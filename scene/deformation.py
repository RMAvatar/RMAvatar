import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from model.smplx_utils import smplx_utils
# from scene.grid import HashHexPlane
from utils.network_util import initseq
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
class Deformation(nn.Module):
    def __init__(self, D=8, W=256,z=72,input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.z = z
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid if hasattr(args, 'no_grid') else False
        self.grid = HexPlaneField(args.bounds if hasattr(args, 'bounds') else 1.6, args.kplanes_config if hasattr(args, 'kplanes_config') else {'grid_dimensions': 2, 'input_coordinate_dim': 4, 'output_coordinate_dim': 32, 'resolution': [64, 64, 64, 75]}, args.multires if hasattr(args, 'multires') else [1, 2])
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel if hasattr(args, 'empty_voxel') else False:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp if hasattr(args, 'static_mlp') else False:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel if hasattr(self.args, 'empty_voxel') else False:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out).cuda()
        # self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)).cuda()
        # self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)).cuda()
        # self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)).cuda()
        # self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)).cuda()
        # self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3)).cuda()
        # self.pos_deform = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(self.W, self.W),
        #     nn.ReLU(),
        #     nn.Linear(self.W, self.W),
        #     nn.ReLU(),
        #     nn.Linear(self.W, self.W),
        #     nn.ReLU(),
        #     nn.Linear(self.W, self.W),
        #     nn.ReLU(),
        #     nn.Linear(self.W, 3),
        #     #nn.Tanh(),
        # ).cuda()

        self.scales_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3),
            #nn.Tanh(),
        ).cuda()

        self.rotations_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 4),
            #nn.Tanh(),
        ).cuda()

        self.opacity_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 1)
        ).cuda()

        self.shs_deform = nn.Sequential(
            nn.Linear(105, 128),#people 105  head 101
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(164, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        ).cuda()

        # self.shs_deform = nn.Sequential(
        #     nn.Linear(105, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(164, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(128, 10)
        # ).cuda()人体的mlp

        self.pos_deform = nn.Sequential(
            nn.Linear(1440, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Tanh(),
        ).cuda()
        self.out1 = nn.Linear(10, 3).cuda()
        self.out2 = nn.Linear(10, 3).cuda()
        self.out3 = nn.Linear(10, 4).cuda()
        # self.dx_min_values = []
        # self.dx_max_values = []
        # self.ds_min_values = []
        # self.ds_max_values = []
        # self.dr_min_values = []
        # self.dr_max_values = []

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])#(x,y,z,t)
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1)
        
        
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        if self.args.static_mlp if hasattr(self.args, 'static_mlp') else False:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel if hasattr(self.args, 'empty_voxel') else False:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
            device = rays_pts_emb.device
            mask = mask.to(device)
        # breakpoint()
        if self.args.no_dx if hasattr(self.args, 'no_dx') else False:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
            # dx_min = dx.min().item()
            # dx_max = dx.max().item()
            # self.dx_min_values.append(dx_min)
            # self.dx_max_values.append(dx_max)
            # print(f"dx min: {dx_min}, dx max: {dx_max}")
        if self.args.no_ds if hasattr(self.args, 'no_ds') else False:
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            # ds_min = ds.min().item()
            # ds_max = ds.max().item()
            # self.ds_min_values.append(ds_min)
            # self.ds_max_values.append(ds_max)
            # print(f"ds min: {ds_min}, ds max: {ds_max}")
            
        if self.args.no_dr if hasattr(self.args, 'no_dr') else False:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = torch.zeros_like(rotations_emb[:,:4])
            # dr_min = dr.min().item()
            # dr_max = dr.max().item()
            # self.dr_min_values.append(dr_min)
            # self.dr_max_values.append(dr_max)
            # print(f"dr min: {dr_min}, dr max: {dr_max}")
            if self.args.apply_rotation if hasattr(self.args, 'apply_rotation') else False:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do if hasattr(self.args, 'no_do') else True:
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs if hasattr(self.args, 'no_dshs') else True:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width if hasattr(args, 'net_width') else 64
        timebase_pe = args.timebase_pe if hasattr(args, 'timebase_pe') else 4
        defor_depth = args.defor_depth if hasattr(args, 'defor_depth') else 0
        posbase_pe = args.posebase_pe if hasattr(args, 'posebase_pe') else 10
        scale_rotation_pe = args.scale_rotation_pe if hasattr(args, 'scale_rotation_pe') else 2
        opacity_pe = args.opacity_pe if hasattr(args, 'opacity_pe') else 2
        timenet_width = args.timenet_width if hasattr(args, 'timenet_width') else 64
        timenet_output = args.timenet_output if hasattr(args, 'timenet_output') else 32
        grid_pe = args.grid_pe if hasattr(args, 'grid_pe') else 0
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.time_poc = self.time_poc.to(device)
        self.pos_poc = self.pos_poc.to(device)
        self.rotation_scaling_poc = self.rotation_scaling_poc.to(device)
        self.opacity_poc = self.opacity_poc.to(device)
        self.apply(initialize_weights)
        # print(self)

    def save_deform_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deformation_net.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_deform_weights(self, model_path, iteration=-1):
        from utils.system_utils import searchForMaxIteration
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, 'deform.pth')# "point_cloud/iteration_{}".format(loaded_iter)
        self.deformation_net.load_state_dict(torch.load(weights_path))

    def forward(self, point, scales=None, rotations=None,pose=None,iteration=None,total_iteration=None):
        return self.forward_dynamic(point, scales, rotations,pose,iteration,total_iteration)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, pose=None,iteration=None,total_iteration=None,scale_offset=None):
        #point_emb = poc_fre(point,self.pos_poc)
        device = point.device
        pose = pose.to(device)
        # point_emb0=torch.cat([pose.unsqueeze(0).repeat(point.shape[0], 1) , point], dim=-1)
        # point_emb = nerf_positional_encoding(point_emb0)

        pos_emb0 = get_embedder(iteration, multires=6, kick_in_iter=1, full_band_iter=total_iteration)[0](point)
        #point_emb = torch.cat([pose.unsqueeze(0).repeat(point.shape[0], 1), pos_emb0], dim=-1)

        if pose.shape[0] == 1:
            point_emb = torch.cat([pose.repeat(point.shape[0], 1), pos_emb0], dim=-1)
        else:
            point_emb = torch.cat([pose.unsqueeze(0).repeat(point.shape[0], 1), pos_emb0], dim=-1)

        for i in range(len(self.deformation_net.shs_deform)):
            if i==4:
                point_emb = torch.cat([point_emb, pos_emb0], dim=-1)
            point_emb = self.deformation_net.shs_deform[i](point_emb)
        offset = point_emb
        means3D = point + offset[..., :3]
        #scales = scales  + offset[..., 3:6]
        if scale_offset == None:
            scales = scales + offset[..., 3:6]
        else:
            scales = torch.log(torch.clamp_min(scales + offset[..., 3:6], 1e-6))
        delta_rot = offset[..., 6:]
        q1 = delta_rot
        q1[:, 0] = 1.
        q2 = rotations
        rotations = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(q1), quat_wxyz_to_xyzw(q2)))
        return means3D, scales, rotations,offset
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)#将input_data、正弦值和余弦值在最后一个维度上拼接起来
    return input_data_emb

def nerf_positional_encoding(input_data, L=10):
    device = input_data.device
    freq_bands = (2.0 ** torch.arange(L, device=device).float()) * torch.pi
    input_data_emb = (input_data.unsqueeze(-1) * freq_bands).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_encoded = torch.cat([input_data_sin, input_data_cos], dim=-1)
    return input_data_encoded

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']#xyz
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']#5
        N_freqs = self.kwargs['num_freqs']#6

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)#2^0-2^5

        # get hann window weights
        kick_in_iter = torch.tensor(self.kwargs['kick_in_iter'],
                                    dtype=torch.float32)
        t = torch.clamp(self.kwargs['iteration'] - kick_in_iter, min=0.)
        N = self.kwargs['full_band_iter'] - kick_in_iter
        m = N_freqs
        alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(iteration,multires,kick_in_iter=0, full_band_iter=50000, is_identity=0):
    if is_identity == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
        'iteration': iteration,
        'kick_in_iter': kick_in_iter,
        'full_band_iter': full_band_iter,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim