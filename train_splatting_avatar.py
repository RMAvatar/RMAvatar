import os
from pathlib import Path
import torch
from gaussian_renderer import network_gui
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from model.splatting_avatar_model import SplattingAvatarModel
from model.splatting_avatar_optim import SplattingAvatarOptimizer
from model.loss_base import run_testing
from dataset.dataset_helper import make_frameset_data, make_dataloader
from model import libcore
from argparse import ArgumentParser, Namespace
from utils.timer import Timer
from utils.metrics import img_mse, img_ssim, img_psnr, perceptual
import lpips
from utils.graphics_utils import compute_face_normals
import torch.nn.functional as F
from utils.general_utils import build_rotation

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

if __name__ == '__main__':
    parser = ArgumentParser(description='SplattingAvatar Training')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], 
                        required=True, help='path to config file')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--deform_on', type=int, default=0)
    args, extras = parser.parse_known_args()

    # output dir创建输出路径
    if args.model_path is None:
        model_path = f"output-splatting/baseline+offset(5w)"
    else:
        model_path = args.model_path

    if not os.path.isabs(model_path):
        model_path = os.path.join(args.dat_dir, model_path)
    os.makedirs(model_path, exist_ok=True)

    # load model and training config
    config = libcore.load_from_config(args.configs, cli_args=extras)#加载config文件
    OmegaConf.save(config, os.path.join(model_path, 'config.yaml'))#将config文件保存到path中
    libcore.set_seed(config.get('seed', 9061))#设置随机数种子

    ##################################################
    config.dataset.dat_dir = args.dat_dir#设置数据目录
    config.cache_dir = os.path.join(args.dat_dir, f'cache_{Path(args.configs[0]).stem}')#设置缓存目录
    frameset_train = make_frameset_data(config.dataset, split='train')#创建训练集帧数据集3646
    frameset_test = make_frameset_data(config.dataset, split='test')#创建测试集帧数据集350
    dataloader = make_dataloader(frameset_train, shuffle=True)#传入训练集帧数据集 frameset_train，并设置 shuffle=True（打乱数据），以创建数据加载器 dataloader

    # first frame as canonical
    first_batch = frameset_train.__getitem__(0)#从 frameset_train 数据集中获取第一个批次的数据
    cano_mesh = first_batch['mesh_info']

    ##################################################
    pipe = config.pipe
    gs_model = SplattingAvatarModel(config.model,cano_mesh,args,verbose=True)
    gs_model.create_from_canonical(cano_mesh)

    gs_optim = SplattingAvatarOptimizer(gs_model, config.optim)

    ##################################################
    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    data_iterator = iter(dataloader)

    total_iteration = config.optim.total_iteration
    first_iter = config.optim.first_iter
    save_every_iter = config.optim.get('save_every_iter', 10000)
    testing_iterations = config.optim.get('testing_iterations', [total_iteration])

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    progress_bar = tqdm(range(first_iter, total_iteration+1), desc="Training progress")

    # 初始化 TensorBoard writer
    tb_writer = SummaryWriter(log_dir=model_path) if TENSORBOARD_FOUND else None

    timer = Timer()
    timer.start()
    first_iter += 1
    for iteration in range(first_iter, total_iteration + 1):
        iter_start.record()
        gs_optim.update_learning_rate(iteration)

    # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gs_model.oneupSHdegree()
 
        try:
            batches = next(data_iterator)
        except:
            data_iterator = iter(dataloader)
            batches = next(data_iterator)

        batch = batches[0]
        frm_idx = batch['frm_idx']
        scene_cameras = batch['scene_cameras']
        pose = batch['mesh_info'].get('pose', None)
        #pose = batch['mesh_info']['pose']

        # mesh_verts = batch['mesh_info']['mesh_verts']
        # mesh_faces = batch['mesh_info']['mesh_faces']
        # def save_mesh_to_obj(mesh_verts, mesh_faces, file_path="output_mesh.obj"):
        #     with open(file_path, 'w') as obj_file:
        #         # Write vertices
        #         for vert in mesh_verts:
        #             obj_file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        #         for face in mesh_faces:
        #             obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
        # save_mesh_to_obj(mesh_verts, mesh_faces, "output_mesh.obj")

        #mesh_normal = compute_face_normals(batch['mesh_info']['mesh_verts'],batch['mesh_info']['mesh_faces'])

        # def save_normals_to_txt(mesh_normal, file_path="normals.txt"):
        #     with open(file_path, 'w') as txt_file:
        #         for normal in mesh_normal:
        #             txt_file.write(f"{normal[0]} {normal[1]} {normal[2]}\n")
        # save_normals_to_txt(mesh_normal, "mesh_normals.txt")

        # update to current posed mesh
        gs_model.update_to_cano_mesh(batch['mesh_info'])
            
        # there should be only one camera
        current_time = timer.get_elapsed_time()
        viewpoint_cam = scene_cameras[0].cuda()
        viewpoint_cam.time = current_time
        gt_image = viewpoint_cam.original_image

        # send one image to gui (optional)将一个图像发送到 GUI（图形用户界面），以便可视化
        if args.ip != 'none':
            network_gui.render_to_network(gs_model, pipe, verify, gt_image=gt_image)

        # render
        white_background=gs_optim.optimizer_config.white_background
        deform_on = bool(args.deform_on)
        render_pkg = gs_model.render_to_camera(viewpoint_cam,pose, pipe,iteration,total_iteration,deform_on,white_background,itr = iteration)
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"],render_pkg["visibility_filter"], render_pkg["radii"]
        image = render_pkg['render']
        gt_image = render_pkg['gt_image']
        if deform_on:
            offset = render_pkg['offset']
        gt_alpha_mask = render_pkg['gt_alpha_mask']

        # from model import libcore
        # output_dir = '/workspace/psen/SplattingAvatar-master/dataset/people/peoplesnapshot/female-3-casual/output-splatting'
        # libcore.write_tensor_image(os.path.join(output_dir, 'render.jpg'), image, rgb2bgr=True)
        # libcore.write_tensor_image(os.path.join(output_dir, 'gt_image.jpg'), gt_image, rgb2bgr=True)

        # rotations = render_pkg['rotations']
        # scales = render_pkg['scales']
        # rotations = rotations / rotations.norm(dim=-1, keepdim=True)
        # normals = F.one_hot(
        #     torch.argmin(scales, dim=-1), num_classes=3
        # ).float()
        # rots = build_rotation(rotations)
        # normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
        # normals = F.normalize(normals, dim=1)
        # normals = normals @ viewpoint_cam.world_view_transform.squeeze(0)[:3, :3]
        # render_normal = torch.nn.functional.normalize(normals, p=2, dim=0)
        # c2w = (viewpoint_cam.world_view_transform.T).inverse()
        # normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        # render_normal_world = normal2.transpose(1, 0)
        # mesh_normal = mesh_normal.to(gs_model.binding.device)[gs_model.binding]

        # loss
        if deform_on:
            loss = gs_optim.collect_loss(image,gt_image,loss_fn_vgg, visibility_filter,viewpoint_cam,tb_writer,iteration,offset,gt_alpha_mask=gt_alpha_mask)
        else:
            loss = gs_optim.collect_loss(image, gt_image, loss_fn_vgg, visibility_filter, viewpoint_cam, tb_writer,iteration, gt_alpha_mask=gt_alpha_mask)
        loss['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss['total'].item() + 0.6 * ema_loss_for_log
            #tb_writer.add_scalar('Loss/ema_loss', ema_loss_for_log, iteration)
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{4}f}"}
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == total_iteration:
                progress_bar.close()

            gs_optim.adaptive_density_control(render_pkg, iteration)

            gs_optim.step()
            gs_optim.zero_grad(set_to_none=True)

        if tb_writer:
            tb_writer.add_scalar('Loss/Total', loss['total'].item(), iteration)
            for key, value in loss.items():
                if key != 'total':
                    tb_writer.add_scalar(f'Loss/{key}', value.item(), iteration)

        # report testing
        #if iteration in testing_iterations:
        test_iterations =  {2000, 5000, 8000,10000,11000,12000,16000,21000,26000,31000,35000,40000,45000,50000,55000,60000}
        #if iteration % 2000 == 0:
        if iteration in test_iterations : # or iteration % 5000 == 0:#10000
            current_time = timer.get_elapsed_time()
            run_testing(current_time,tb_writer,pipe, frameset_test, gs_model,deform_on,white_background, model_path, iteration,total_iteration, verify=verify)

        # save
        save_iterations = {2000, 5000, 8000,11000,16000,21000,26000,31000,35000,40000,45000,50000,55000,60000}
        #if iteration % save_every_iter == 0:
        if iteration in save_iterations : #or iteration % 5000 == 0:
            pc_dir = gs_optim.save_checkpoint(model_path, iteration)
            libcore.write_tensor_image(os.path.join(pc_dir, 'gt_image.jpg'), gt_image, rgb2bgr=True)
            libcore.write_tensor_image(os.path.join(pc_dir, 'render.jpg'), image, rgb2bgr=True)

            _rmse = img_mse(image[None, ...], gt_image[None, ...], mask=None, error_type='rmse', use_mask=False)
            psnr = img_psnr(image[None, ...], gt_image[None, ...], rmse=_rmse)
            if tb_writer:
                tb_writer.add_scalar('PSNR', psnr.item(), iteration)

    ##################################################
    # training finished. hold on
    while network_gui.conn is not None:
        network_gui.render_to_network(gs_model, pipe, args.dat_dir)

    if tb_writer:
        tb_writer.close()

    print('[done]')