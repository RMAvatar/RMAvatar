# RMAvatar

## Getting Started
- Create conda env with pytorch.
```
conda create -n rmavatar python=3.10
conda activate rmavatar

# pytorch 2.0.1+cu117 is tested
pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117

# install other dependencies
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
pip install .
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
cd simple-knn
pip install .
pip install -r requirements.txt
```
## Human Avatar
We conducted experiments on [PeopleSnapshot](https://graphics.tu-bs.de/people-snapshot).
- Please download the parameter files (the same with Anim-NeRF) from: [Baidu Disk](https://pan.baidu.com/s/1CSi2iujDb2vd6pWkRaJsTw?pwd=is8s)
- Download 4 sequences from PeopleSnapshot (male/female-3/4-casual) and unzip `images` and `masks` to corresponding folders from above.
- Use `scripts/preprocess_PeopleSnapshot.py` to preprocess the data.
- Training:
```
python train_splatting_avatar.py --config configs/splatting_avatar.yaml;configs/instant_avatar.yaml --dat_dir <path/to/subject> --deform_on 1 --model_path <path/to/subject/output-splatting/> 
# to animate to noval pose `aist_demo.npz`
python eval_animate.py --config "configs/splatting_avatar.yaml;configs/instant_avatar.yaml" --dat_dir /path-to/female-3-casual --pc_dir /path-to/female-3-casual/output-splatting/last_checkpoint/point_cloud/iteration_50000 --anim_fn /path-to/aist_demo.npz
```

## Citation
If you find our code or paper useful, please cite as:
```
@article{peng2025rmavatar,
  title={RMAvatar: Photorealistic Human Avatar Reconstruction from Monocular Video Based on Rectified Mesh-embedded Gaussians},
  author={Peng, Sen and Xie, Weixing and Wang, Zilong and Guo, Xiaohu and Chen, Zhonggui and Yang, Baorong and Dong, Xiao},
  journal={arXiv preprint arXiv:2501.07104},
  year={2025}
}
```

## Acknowledgement
We thank the following authors for their excellent works!
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
- [SplattingAvatar](https://github.com/initialneil/SplattingAvatar)
- [Anim-Nerf](https://github.com/JanaldoChen/Anim-NeRF)
- [InstantAvatar](https://github.com/tijiang13/InstantAvatar)

## License
RMAvatar
<br>
The code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for Noncommercial use only. Any commercial use should get formal permission first.

[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)
<br>
**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**. The *Software* is in the process of being registered with the Agence pour la Protection des Programmes (APP).  
