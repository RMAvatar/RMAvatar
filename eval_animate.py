import os
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from model.splatting_avatar_model import SplattingAvatarModel
from model.loss_base import run_testing
from dataset.dataset_helper import make_frameset_data, make_dataloader
from scene.dataset_readers import make_scene_camera
from gaussian_renderer import network_gui
from model import libcore
from tqdm import tqdm


# altered from InstantAvatar
# https://github.com/tijiang13/InstantAvatar/blob/master/animate.py
class AnimateDataset(torch.utils.data.Dataset):
    def __init__(self, pose_sequence, betas):
        smpl_params = dict(np.load(pose_sequence))

        thetas = smpl_params["poses"][..., :72]
        transl = smpl_params["trans"] - smpl_params["trans"][0:1]
        transl += (0, 0.15, 5)

        self.betas = betas
        self.thetas = torch.tensor(thetas).float()
        self.transl = torch.tensor(transl).float()

    def __len__(self):
        return len(self.transl)

    def __getitem__(self, idx):
        datum = {
            # SMPL parameters
            "betas": self.betas,
            "global_orient": self.thetas[idx:idx + 1, :3],
            "body_pose": self.thetas[idx:idx + 1, 3:],
            "transl": self.transl[idx:idx + 1],
        }
        return datum


##################################################
if __name__ == '__main__':
    parser = ArgumentParser(description='SplattingAvatar Evaluation')
    parser.add_argument('--ip', type=str, default='127.0.0.2')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')],
                        required=True, help='path to config file')
    parser.add_argument('--pc_dir', type=str, default=None)
    parser.add_argument('--total_iteration', type=int, default=50000)
    parser.add_argument('--anim_fn', type=str, required=True)
    parser.add_argument('--deform_on', type=int, default=0)
    parser.add_argument('--white_background', type=bool, default=True)
    args, extras = parser.parse_known_args()

    # load model and training config
    config = libcore.load_from_config(args.configs, cli_args=extras)

    ##################################################
    config.dataset.dat_dir = args.dat_dir
    frameset_train = make_frameset_data(config.dataset, split='train')

    smpl_model = frameset_train.smpl_model
    cam = frameset_train.cam
    empty_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
    viewpoint_cam = make_scene_camera(0, cam, empty_img)
    mesh_py3d = frameset_train.mesh_py3d

    # anim
    betas = frameset_train.smpl_params['betas']
    anim_data = AnimateDataset(args.anim_fn, betas)

    # output dir
    subject = Path(args.dat_dir).stem
    out_dir = os.path.join(Path(args.anim_fn).parent, f'anim/anim_{subject}_final_on')
    os.makedirs(out_dir, exist_ok=True)

    ##################################################
    first_batch = frameset_train.__getitem__(0)
    cano_mesh = first_batch['mesh_info']
    pipe = config.pipe

    #binding_path = os.path.join(args.pc_dir, 'binding.pkl')
    gs_model = SplattingAvatarModel(config.model, cano_mesh, args, verbose=True)
    ply_fn = os.path.join(args.pc_dir, 'point_cloud.ply')
    gs_model.load_ply(ply_fn)
    deform_fn = args.pc_dir
    gs_model.load_deform_weights(deform_fn,iteration =args.total_iteration)
    embed_fn = os.path.join(args.pc_dir, 'embedding.json')
    gs_model.load_from_embedding(embed_fn)

    ##################################################
    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    ##################################################
    for idx in tqdm(range(len(anim_data))):
        pose_params = anim_data.__getitem__(idx)
        out = smpl_model(**pose_params)
        frame_mesh = mesh_py3d.update_padded(out['vertices'])
        mesh_info = {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }
        # def save_obj(filename, verts, faces):
        #     with open(filename, 'w') as f:
        #         for vert in verts:
        #             f.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))  # 写入顶点坐标
        #         for face in faces:
        #             f.write("f")
        #             for idx in face:
        #                 f.write(" {}".format(idx + 1))  # 索引从1开始
        #             f.write("\n")
        # save_obj('smpl392.obj',mesh_info['mesh_verts'], mesh_info['mesh_faces'])

        pose = out['body_pose'].squeeze()

        gs_model.update_to_cano_mesh(mesh_info)

        from utils.timer import Timer

        timer = Timer()
        timer.start()
        current_time = timer.get_elapsed_time()
        viewpoint_cam.time = current_time

        deform_on = bool(args.deform_on)
        render_pkg = gs_model.render_to_camera(viewpoint_cam,pose, pipe,args.total_iteration,args.total_iteration,deform_on,args.white_background)
        #gs_model.save_ply(os.path.join(out_dir, 'point_cloud.ply'))
        image = render_pkg['render']

        if verify is not None:
            network_gui.send_image_to_network(image, verify)

        libcore.write_tensor_image(os.path.join(out_dir, f'{idx:04d}.jpg'), image, rgb2bgr=True)

    print('[done]')