# IMavatar data Reader.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import cv2
import json
from copy import deepcopy
import torch
import numpy as np
from scene.dataset_readers import convert_to_scene_cameras
from model import libcore
from model.imavatar.flame import FLAME
import pytorch3d.structures.meshes as py3d_meshes

def read_imavatar_frameset(dat_dir, frame_info, intrinsics, extension='.png'):
    w2c = np.array(frame_info['world_mat'])#获取世界坐标系到相机坐标系的变换矩阵
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)#（3,4）>(4,4)

    # standard R, t in OpenCV coordinate
    R = w2c[:3,:3]#旋转矩阵
    T = w2c[:3, 3]#平移向量
    R[1:, :] = -R[1:, :]
    T[1:] = -T[1:]#修正坐标系之间的差异

    # Note:
    # R is stored transposed (R.T) due to 'glm''s column-major storage in CUDA code

    # dirty fix
    file_path = frame_info['file_path']
    file_path = file_path.replace('/image/', '/images/')

    image_path = os.path.abspath(os.path.join(dat_dir, file_path + extension)).replace('\\', '/')
    if not os.path.exists(image_path):
        image_path = image_path.replace('/images/', '/image/')

    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)#读取第一张图片

    cam = libcore.Camera()
    cam.h, cam.w = image.shape[:2]#将相机的高度和宽度设置为图像的高度和宽度
    cam.fx = abs(cam.w * intrinsics[0])
    cam.fy = abs(cam.h * intrinsics[1])
    cam.cx = abs(cam.w * intrinsics[2])
    cam.cy = abs(cam.h * intrinsics[3])#计算相机的焦距和光心位置
    cam.R = R
    cam.setTranslation(T)#设置相机的旋转矩阵和平移向量
    # print(cam)

    color_frames = libcore.DataVec()
    color_frames.cams = [cam]
    color_frames.frames = [image]
    color_frames.images_path = [image_path]
    return color_frames#包含相机、图片和图像路径信息

class IMavatarDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split

        self.dat_dir = config.dat_dir
        self.cameras_extent = config.get('cameras_extent', 1.0)

        self.num_for_train = config.get('num_for_train', -350)

        self.load_flame_json()
        self.num_frames = len(self.frm_list)
        print(f'[IMavatarDataset][{self.split}] num_frames = {self.num_frames}')

    ##################################################
    # load flame_params.json
    def load_flame_json(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'flame_params.json')):#frames_info
            raise NotImplementedError
        
        with open(os.path.join(self.dat_dir, 'flame_params.json')) as fp:
            contents = json.load(fp)
            self.intrinsics = contents['intrinsics']
            self.shape_params = torch.tensor(contents['shape_params']).unsqueeze(0)

            if self.split == 'train':
                self.frames_info = contents['frames'][:self.num_for_train]
            else:
                self.frames_info = contents['frames'][self.num_for_train:]#350张图片作为测试集

            self.frm_list = []
            self.flame_params = []
            for frame in self.frames_info:
                frm_idx = os.path.basename(frame['file_path'])
                self.frm_list.append(frm_idx)
                self.flame_params.append({
                    'full_pose': torch.tensor(frame['pose']).unsqueeze(0),
                    'expression_params': torch.tensor(frame['expression']).unsqueeze(0),
                })#读取每一帧的位姿和表情

        """
        - This is the IMAvatar/DECA version of FLAME
        - Originally from: https://github.com/zhengyuf/IMavatar/tree/main/code/flame
        - What's changed from normal FLAME:
          - There's a `factor=4` in the `flame.py`, making the output mesh 4 times larger
          - The input `full_pose` is [Nx15], which is a combination of different pose components
          - In a standard FLAME model, there is `pose_params`[Nx6], `neck_pose`[Nx3], `eye_pose`[Nx6].
            To convert to `full_pose`:
            ```
            # [3] global orient
            # [3] neck
            # [3] jaw
            # [6] eye
            full_pose = torch.concat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=-1)
            ```
        """
        self.flame = FLAME('model/imavatar/FLAME2020/flame2023.pkl',
                           'model/imavatar/FLAME2020/landmark_embedding.npy',
                           n_shape=100,
                           n_exp=50,
                           shape_params=self.shape_params,
                           canonical_expression=None,
                           canonical_pose=None)
        self.mesh_py3d = py3d_meshes.Meshes(self.flame.v_template[None, ...].float(), 
                                            torch.from_numpy(self.flame.faces[None, ...].astype(int)))#使用PyTorch3D传入顶点和面数据得到mesh

    ##################################################
    def __len__(self):
        return len(self.frm_list)#3646

    def __getitem__(self, idx):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = int(idx)

        # frames
        color_frames = read_imavatar_frameset(self.dat_dir, self.frames_info[idx], self.intrinsics)#包含相机、图片和图像路径信息
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)#将给定的color_frames数据转换为一个包含场景相机对象的列表
        
        batch = {
            'idx': idx,#3646
            'frm_idx': frm_idx,
            'color_frames': color_frames,#包含相机、图片和图像路径信息
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        }

        # mesh
        batch['mesh_info'] = self.get_flame_mesh(idx)
        return batch#获取FLAME网格信息，更新batch

    def get_flame_mesh(self, idx):
        with torch.no_grad():
            flame_params = self.get_flame_params(idx)#获取flame的参数
            vertices, _, _ = self.flame(flame_params['expression_params'], flame_params['full_pose'])#计算 FLAME 模型的顶点位置
            pose = torch.cat([flame_params['expression_params'], flame_params['full_pose']], dim=1)

            frame_mesh = self.mesh_py3d.update_padded(vertices)
            # mesh_verts = frame_mesh.verts_packed()  # 更新后的网格的顶点坐标
            # mesh_faces = frame_mesh.faces_packed()  # 更新后的网格的面
            # self.save_obj('bala.obj', mesh_verts, mesh_faces)  # 保存OBJ文件

        return {
            'mesh_verts': frame_mesh.verts_packed(),#更新后的网格的顶点坐标
            'mesh_norms': frame_mesh.verts_normals_packed(),#更新后网格的顶点法线
            'mesh_faces': frame_mesh.faces_packed(),
            'pose': pose,
        }

    # def save_obj(self, filename, verts, faces):
    #     with open(filename, 'w') as f:
    #         for vert in verts:
    #             f.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))  # 写入顶点坐标
    #         for face in faces:
    #             f.write("f")
    #             for idx in face:
    #                 f.write(" {}".format(idx + 1))  # 索引从1开始
    #             f.write("\n")


    def get_flame_params(self, idx):
        return {
            'n_shape': 100,#形状数量
            'n_exp': 50,#表情数量
            'shape_params': self.shape_params,
            **self.flame_params[idx],
        }
    

    