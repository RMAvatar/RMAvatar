# SplattingAvatar Model.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import torch
import torch.nn.functional as thf
import numpy as np
from torch import nn
from pathlib import Path
import json
from model import libcore
#from simple_phongsurf import PhongSurfacePy3d
from utils.sh_utils import eval_sh, RGB2SH
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.data_utils import sample_bary_on_triangles, retrieve_verts_barycentric
from utils.map import PerVertQuaternion
from utils.graphics_utils import BasicPointCloud
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_multiply
from .gauss_base import GaussianBase, to_abs_path, to_cache_path
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from scene.deformation import deform_network
from utils.timer import Timer

# standard 3dgs
class SplattingAvatarModel(GaussianBase):
    def __init__(self, config,cano_mesh,args,
                 device=torch.device('cuda'),
                 verbose=False):
        super().__init__(args)
        self.config = config
        self.device = device
        self.verbose = verbose
        self._deformation = deform_network(args)

        self.register_buffer('_xyz', torch.Tensor(0))
        self.register_buffer('_features_dc', torch.Tensor(0))
        self.register_buffer('_features_rest', torch.Tensor(0))
        self.register_buffer('_scaling', torch.Tensor(0))
        self.register_buffer('_rotation', torch.Tensor(0))
        self.register_buffer('_opacity', torch.Tensor(0))
        self.register_buffer('_mask', torch.Tensor(0))

        # for splatting avatar
        self.register_buffer('max_radii2D', torch.Tensor(0))
        self.register_buffer('xyz_gradient_accum', torch.Tensor(0))
        self.register_buffer('denom', torch.Tensor(0))

        #self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.face_center = None
        self.face_scaling = None
        self.face_orien_mat = None
        self.face_orien_quat = None
        self.binding = None
        self.binding_counter = None
        self.timestep = None
        self.num_timesteps = 1

        if config is not None:
            self.setup_config(config)

        cano_faces = cano_mesh['mesh_faces'].long().to(self.device)

        # import pickle
        # if binding_path is not None and os.path.exists(binding_path):
        #     with open(binding_path, 'rb') as binding_file:
        #         self.binding = pickle.load(binding_file)
        #     # Initialize binding_counter based on loaded binding
        #     self.binding_counter = torch.ones(len(self.binding), dtype=torch.int32).to(self.device)
        #     print(f'[Binding] Loaded binding from {binding_path}')
        # else:
        #     self.binding = torch.arange(len(cano_faces)).to(self.device)
        #     self.binding_counter = torch.ones(len(cano_faces), dtype=torch.int32).to(self.device)

        if self.binding is None:
            self.binding = torch.arange(len(cano_faces), device=self.device)
            self.binding_counter = torch.ones(len(cano_faces), dtype=torch.int32,device=self.device)

    ##################################################

    def save_deform_weights(self, model_path, iteration):
        self._deformation.save_deform_weights(model_path, iteration)
    def load_deform_weights(self, model_path, iteration):
        self._deformation.load_deform_weights(model_path, iteration)

    @property
    def num_gauss(self):
        return self._xyz.shape[0]

    @property
    def get_xyz_cano(self):
        if self.config.xyz_as_uvd:
            # uv -> self.sample_bary -> self.base_normal --(d)--> xyz
            xyz = self.base_normal_cano * self._xyz[..., -1:]
            return self.base_xyz_cano + xyz
        else:
            return self._xyz

    @property
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def get_covariance_deform(self, scales, rotations, scaling_modifier = 1):
        return self.covariance_activation(scales, scaling_modifier, rotations)

    @property
    def get_xyz(self):
        if self.binding is None:
            return self._xyz
        else:
             xyz = torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1)
             return xyz * self.face_scaling[self.binding] + self.face_center[self.binding]
        # if self.config.xyz_as_uvd:
        #     # uv -> self.sample_bary -> self.base_normal --(d)--> xyz
        #     xyz = self.base_normal * self._xyz[..., -1:]
        #     return self.base_xyz + xyz
        # else:
        #     return self._xyz

    @property
    def base_normal_cano(self):
        return thf.normalize(retrieve_verts_barycentric(self.cano_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)

    @property
    def base_normal(self):
        return thf.normalize(retrieve_verts_barycentric(self.mesh_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)
    @property
    def base_xyz_cano(self):
        return retrieve_verts_barycentric(self.cano_verts, self.cano_faces, 
                                          self.sample_fidxs, self.sample_bary)

    @property
    def base_xyz(self):
        return retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, 
                                          self.sample_fidxs, self.sample_bary)

    @property
    def get_rotation_cano(self):
        return self.rotation_activation(self._rotation)
        
    @property
    def get_rotation(self):
        if self.binding is None:
            return self.rotation_activation(self._rotation)
        else:
            rot = self.rotation_activation(self._rotation)
            face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
            return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))
        #return self.rotation_activation(quaternion_multiply(self.base_quat, self._rotation))

        #q1_loaded = torch.load('/workspace/psen/SplattingAvatar-master/dataset/q1.pt', map_location='cuda')
        #q2_loaded = torch.load('/workspace/psen/SplattingAvatar-master/dataset/q2.pt', map_location='cuda')
        #q1_tensor = torch.tensor(q1_loaded)
        #q2_tensor = torch.tensor(q2_loaded)
        #quat_product_result = quat_product(q1_tensor, q2_tensor)
        #a=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(q1_tensor), quat_wxyz_to_xyzw(q2_tensor)))



    
    @property
    def get_rotation_embed(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def base_quat(self):
        return torch.einsum('bij,bi->bj', self.tri_quats[self.sample_fidxs], self.sample_bary)
    
    @property
    def get_scaling_cano(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling(self):
        if self.binding is None:
            return self.scaling_activation(self._scaling)
        else:
            scaling = self.scaling_activation(self._scaling)
            return scaling * self.face_scaling[self.binding]
        # if self.config.with_mesh_scaling:
        #     scaling_alter = self._face_scaling[self.sample_fidxs]
        #     return self.scaling_activation(self._scaling * scaling_alter)
        # else:
        #     return self.scaling_activation(self._scaling)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def get_params(self, device='cpu'):
        return {
            '_xyz': self._xyz.detach().to(device),
            '_rotation': self._rotation.detach().to(device),
            '_scaling': self._scaling.detach().to(device),
            '_features_dc': self._features_dc.detach().to(device),
            '_features_rest': self._features_rest.detach().to(device),
            '_opacity': self._opacity.detach().to(device),
        }
    
    def set_params(self, params):
        if '_xyz' in params:
            self._xyz = params['_xyz'].to(self.device)
        if '_rotation' in params:
            self._rotation = params['_rotation'].to(self.device)
        if '_scaling' in params:
            self._scaling = params['_scaling'].to(self.device)
        if '_features_dc' in params:
            self._features_dc = params['_features_dc'].to(self.device)
        if '_features_rest' in params:
            self._features_rest = params['_features_rest'].to(self.device)
        if '_opacity' in params:
            self._opacity = params['_opacity'].to(self.device)
    
    def get_colors_precomp(self, viewpoint_camera=None):
        return self.color_activation(self._color)
    
    def get_colors_precomp(self, viewpoint_camera=None):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        if viewpoint_camera is not None:
            dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        else:
            dir_pp_normalized = torch.zeros_like(self._xyz)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp

    def contract_to_unisphere(self,
                              x: torch.Tensor,
                              aabb: torch.Tensor,
                              ord: int = 2,
                              eps: float = 1e-6,
                              derivative: bool = False,
                              ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag ** 2 + 2 * x ** 2 * (
                    1 / mag ** 3 - (2 * mag - 1) / mag ** 4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x


    ##################################################
    def setup_config(self, config):
        self.config = config
        self.max_sh_degree = config.get('sh_degree', 0)

        # use _xyz as variables for uvd
        # enabling uvd representation of SplattingAvatar
        self.config.xyz_as_uvd = self.config.get('xyz_as_uvd', True)
        self.config.with_mesh_scaling = config.get('with_mesh_scaling', False)

    def create_from_pcd(self, pcd : BasicPointCloud):
        num_pts = self.binding.shape[0]
        fused_point_cloud = torch.zeros((num_pts, 3), device=self.device)
        #fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))#将pcd.colors 转换为球谐函数表示的颜色
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 梯度追踪
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)), 0.0000001)#distCUDA2计算每个点到其最近k点的平均距离平方，torch.clamp_min确保这些距离平方的最小值不小于 0.0000001
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)#log根号每个点的距离平方
        scales = torch.log(torch.ones((num_pts, 3), device="cuda"))
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1#将所有四元数的实数部分设置为 1。因为四元数的实数部分表示旋转的幅度，虚数部分表示旋转的轴向分量，因此将实数部分设为 1 而虚数部分保持为 0 表示没有旋转

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # self._xyz = fused_point_cloud
        # self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        # self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        # self._scaling = scales
        # self._rotation = rots
        # self._opacity = opacities
        # self.max_radii2D = torch.zeros((num_pts), device="cuda")
        self.active_sh_degree = self.max_sh_degree  #0
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((num_pts), device="cuda")
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
    def setup_canonical(self, cano_verts, cano_norms, cano_faces):
        self.cano_verts = cano_verts
        self.cano_norms = cano_norms
        self.cano_faces = cano_faces

        # quaternion from cano to pose
        #self.quat_helper = PerVertQuaternion(cano_verts, cano_faces).to(self.device)#包含三角形的面积

        # phong surface for triangle walk
        # self.phongsurf = PhongSurfacePy3d(cano_verts, cano_faces, cano_norms,
        #                                   outer_loop=2, inner_loop=50, method='uvd').to(self.device)#初始化三角形的行走

    def create_from_canonical(self, cano_mesh, sample_fidxs=None, sample_bary=None):
        cano_verts = cano_mesh['mesh_verts'].float().to(self.device)
        cano_norms = cano_mesh['mesh_norms'].float().to(self.device)
        cano_faces = cano_mesh['mesh_faces'].long().to(self.device)
        xyz_max = cano_verts.max(dim=0).values
        xyz_min = cano_verts.min(dim=0).values
        xyz_max = xyz_max.tolist()
        xyz_min = xyz_min.tolist()
        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        self.setup_canonical(cano_verts, cano_norms, cano_faces)#将这些参数传到gpu上面并且初始化三角形的行走
        self.mesh_verts = self.cano_verts
        self.mesh_norms = self.cano_norms
        
        # # sample on mesh
        # if sample_fidxs is None or sample_bary is None:
        #     num_samples = self.config.get('num_init_samples', 10000)#获取采样点的数量，默认为 10000
        #     sample_fidxs, sample_bary = sample_bary_on_triangles(cano_faces.shape[0], num_samples)#在三角形网格上进行采样，得到采样点的索引和重心坐标。
        # self.sample_fidxs = sample_fidxs.to(self.device)
        # self.sample_bary = sample_bary.to(self.device)
        #
        # sample_verts = retrieve_verts_barycentric(cano_verts, cano_faces, self.sample_fidxs, self.sample_bary)#通过重心坐标计算采样点的实际顶点位置
        # sample_norms = retrieve_verts_barycentric(cano_norms, cano_faces, self.sample_fidxs, self.sample_bary)#通过重心坐标计算采样点的法线
        # sample_norms = thf.normalize(sample_norms, dim=-1)#对发现进行归一化处理

        # pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(),
        #                       normals=sample_norms.detach().cpu().numpy(),
        #                       colors=torch.full_like(sample_verts, 0.5).float().cpu())
        pcd = BasicPointCloud(points=self.mesh_verts.detach().cpu().numpy(),
                              normals=self.mesh_norms.detach().cpu().numpy(),
                              colors=torch.full((cano_faces.shape[0], 3), 0.5).float().cpu())#colors=torch.full((13776, 3), 0.5).float().cpu())
        #colors=torch.full_like(self.mesh_verts, 0.5).float().cpu())
        self.create_from_pcd(pcd)#根据点云数据创建三维模型

        # # use _xyz as uvd
        # if self.config.xyz_as_uvd:
        #     self._xyz = torch.zeros_like(self._xyz)

    # def update_to_cano_mesh(self,mesh_info):
    #     posed_verts = mesh_info['mesh_verts']
    #     posed_faces = mesh_info['mesh_faces']
    #     # self.setup_canonical(cano_verts, cano_norms, cano_faces)  # 将这些参数传到gpu上面并且初始化三角形的行走
    #     # self.mesh_verts = self.cano_verts
    #     # self.mesh_norms = self.cano_norms
    #     # self.mesh_faces =self.cano_faces
    #     triangles = posed_verts[:, posed_faces]
    #     #triangles = self.mesh_verts[self.mesh_faces]
    #     self.face_center = triangles.mean(dim=-2).squeeze(0)
    #     self.face_orien_mat, self.face_scaling = compute_face_orientation(posed_verts.squeeze(0), posed_faces.squeeze(0),return_scale=True)
    #     self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))

    def update_to_cano_mesh(self, posed_mesh=None):
        if posed_mesh is not None:
            self.mesh_verts = posed_mesh['mesh_verts'].unsqueeze(0).float()
            self.mesh_faces = posed_mesh['mesh_faces']
            triangles = self.mesh_verts[:, self.mesh_faces]
            self.face_center = triangles.mean(dim=-2).squeeze(0).to(self.device)
            self.face_orien_mat, self.face_scaling= compute_face_orientation(self.mesh_verts.squeeze(0),self.mesh_faces.squeeze(0),return_scale=True)
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))
            self.face_orien_mat =  self.face_orien_mat.to(self.device)
            self.face_scaling =  self.face_scaling.to(self.device)
            self.face_orien_quat = self.face_orien_quat.to(self.device)
            # self.per_vert_quat = self.quat_helper(self.mesh_verts)
            # self.tri_quats = self.per_vert_quat[self.cano_faces]

        #self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)

    def save_obj(self, filename, verts, faces):
        with open(filename, 'w') as f:
            for vert in verts:
                f.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))  # 写入顶点坐标
            for face in faces:
                f.write("f")
                for idx in face:
                    f.write(" {}".format(idx + 1))  # 索引从1开始
                f.write("\n")

    def update_to_canomesh(self):
        cano = {
            'mesh_verts': self.cano_verts,
            'mesh_norms': self.cano_norms,
            'mesh_faces': self.cano_faces,
        }
        self.update_to_cano_mesh(cano)

    ##################################################
    def prune_points(self, valid_points_mask, optimizable_tensors):
        self._xyz = optimizable_tensors['_xyz']

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = self._scaling[valid_points_mask]

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = self._rotation[valid_points_mask]

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        # if self.config.xyz_as_uvd:
        #     self.sample_fidxs = self.sample_fidxs[valid_points_mask]
        #     self.sample_bary = self.sample_bary[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, optimizable_tensors, densify_out):
        self._xyz = optimizable_tensors.get('_xyz', self._xyz)

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = torch.cat([self._scaling, densify_out['new_scaling']], dim=0)

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = torch.cat([self._rotation, densify_out['new_rotation']], dim=0)

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        # mesh embedding
        # if self.config.xyz_as_uvd:
        #     self.sample_fidxs = torch.cat([self.sample_fidxs, densify_out['new_sample_fidxs']], dim=0)
        #     self.sample_bary = torch.cat([self.sample_bary, densify_out['new_sample_bary']], dim=0)

        # stats
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

    def prepare_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()#处理梯度张量的尺寸与初始点数不一致
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        if self.config.get('force_scaling_split', False):
            aspect_mask = (torch.max(self.get_scaling, dim=-1).values / self.get_scaling.mean(dim=-1)) > 2.0
            force_mask = torch.max(self.get_scaling, dim=-1).values > self.percent_dense * scene_extent * 1
            force_mask = torch.logical_and(force_mask, aspect_mask)
            selected_pts_mask = torch.logical_or(selected_pts_mask, force_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)#筛选出的点的尺度作为标准差
        means = torch.zeros((stds.size(0), 3), device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        return selected_pts_mask, new_xyz.detach()

    def prepare_split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        if self.binding is not None:
            selected_scaling = self.get_scaling[selected_pts_mask]
            face_scaling = self.face_scaling[self.binding[selected_pts_mask]]
            new_scaling = self.scaling_inverse_activation((selected_scaling / face_scaling).repeat(N, 1) / (0.8 * N))
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask].repeat(N)
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding,
                                                       torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))

        # new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        splitout = {
            'new_xyz': new_xyz,
            'new_scaling': new_scaling,
            'new_rotation': new_rotation,
        }

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        splitout.update({
            'new_features_dc': new_features_dc,
            'new_features_rest': new_features_rest,
            'new_opacity': new_opacity,
        })

        return splitout

    def prepare_densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        cloneout = {
            'new_xyz': new_xyz,
            'new_scaling': new_scaling,
            'new_rotation': new_rotation,
        }

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask]#筛选出符合条件的点
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding,
                                                       torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))#13776+1452
        cloneout.update({
            'new_features_dc': new_features_dc,
            'new_features_rest': new_features_rest,
            'new_opacity': new_opacity,
        })

        return cloneout

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    ########################################
    def walking_on_triangles(self):
        if self.config.get('skip_triangle_walk', False):
            return

        fidx = self.sample_fidxs.detach().cpu().numpy().astype(np.int32)#当前高斯点所在的三角形索引
        uv = self.sample_bary[..., :2].detach().cpu().numpy().astype(np.double)#包含当前所有高斯点的重心坐标
        delta = self._xyz[..., :2].detach().cpu().numpy().astype(np.double)
        fidx, uv = self.phongsurf.triwalk.updateSurfacePoints(fidx, uv, delta)

        self.sample_fidxs = torch.tensor(fidx).long().to(self.device)
        self.sample_bary[..., :2] = torch.tensor(uv).float().to(self.device)
        self.sample_bary[..., 2] = 1.0 - self.sample_bary[..., 0] - self.sample_bary[..., 1]

    ########################################
    def load_from_embedding(self, embed_fn):
        with open(embed_fn, 'r') as fp:
            cc = json.load(fp)

        mesh_fn = cc['cano_mesh']
        if not os.path.isabs(mesh_fn):
            mesh_fn = str(Path(embed_fn).parent / mesh_fn)
        
        cano_mesh = libcore.MeshCpu(mesh_fn)
        cano_verts = torch.tensor(cano_mesh.V).float().to(self.device)
        cano_norms = torch.tensor(cano_mesh.N).float().to(self.device)
        cano_faces = torch.tensor(cano_mesh.F).long().to(self.device)
        self.setup_canonical(cano_verts, cano_norms, cano_faces)

        self._xyz = torch.tensor(cc['_xyz']).float().to(self.device)
        if '_rotation' in cc:
            self._rotation = torch.tensor(cc['_rotation']).float().to(self.device)
        # self.sample_fidxs = torch.tensor(cc['sample_fidxs']).long().to(self.device)
        # self.sample_bary = torch.tensor(cc['sample_bary']).float().to(self.device)

    def load_from_embedding_v2(self, embed_fn):
        with open(embed_fn, 'r') as fp:
            cc = json.load(fp)

        mesh_fn = cc['mesh_fn']
        if not os.path.isabs(mesh_fn):
            mesh_fn = str(Path(embed_fn).parent / mesh_fn)
        
        cano_mesh = libcore.MeshCpu(mesh_fn)
        cano_verts = torch.tensor(cano_mesh.V).float().to(self.device)
        cano_norms = torch.tensor(cano_mesh.N).float().to(self.device)
        cano_faces = torch.tensor(cano_mesh.F).long().to(self.device)
        self.setup_canonical(cano_verts, cano_norms, cano_faces)

        self._xyz = torch.tensor(cc['_xyz']).float().to(self.device)
        self._rotation = torch.tensor(cc['_rotation']).float().to(self.device)
        self.sample_fidxs = torch.tensor(cc['sample_fidxs']).long().to(self.device)
        self.sample_bary = torch.tensor(cc['_sample_bary']).float().to(self.device)

        self.save_embedding_json(embed_fn.replace('canonical.sample.json', 'embedding.json'))
        
    def save_embedding_json(self, embed_fn):
        obj_fn = embed_fn.replace('.json', '.obj')
        cano_mesh = libcore.MeshCpu()
        cano_mesh.V = self.cano_verts.detach().cpu()
        cano_mesh.N = self.cano_norms.detach().cpu()
        cano_mesh.F = self.cano_faces.detach().cpu()
        cano_mesh.FN = cano_mesh.F
        cano_mesh.save_to_obj(obj_fn)

        embedding = {
            'cano_mesh': Path(obj_fn).name,
            #'sample_fidxs': self.sample_fidxs.detach().cpu().tolist(),
            #'sample_bary': self.sample_bary.detach().cpu().tolist(),
            '_xyz': self._xyz.detach().cpu().tolist(),
            '_rotation': self._rotation.detach().cpu().tolist(),
        }

        with open(os.path.join(embed_fn), 'w') as f:
            json.dump(embedding, f)

    def save_pose_mesh(self, embed_fn,mesh_verts):
        obj_fn = embed_fn.replace('.json', '.obj')
        cano_mesh = libcore.MeshCpu()
        cano_mesh.V = mesh_verts.detach().cpu()
        cano_mesh.N = self.cano_norms.detach().cpu()
        cano_mesh.F = self.cano_faces.detach().cpu()
        cano_mesh.FN = cano_mesh.F
        cano_mesh.save_to_obj(obj_fn)

        embedding = {
            'cano_mesh': Path(obj_fn).name,
            #'sample_fidxs': self.sample_fidxs.detach().cpu().tolist(),
            #'sample_bary': self.sample_bary.detach().cpu().tolist(),
            '_xyz': self._xyz.detach().cpu().tolist(),
            '_rotation': self._rotation.detach().cpu().tolist(),
        }

        with open(os.path.join(embed_fn), 'w') as f:
            json.dump(embedding, f)