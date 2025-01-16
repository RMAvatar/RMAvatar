# RMAvatar

Getting Started

Create conda env with pytorch.

conda create -n splatting python=3.10
conda activate rmavatar

# pytorch 2.0.1+cu117 is tested

# install other dependencies
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
pip install .

git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
cd simple-knn
pip install .

pip install -r requirements.txt

Preparing dataset
Our preprocessing followed Animatable Neural Radiance Fields from Monocular RGB Videos.

Training
python train_splatting_avatar.py --config configs/splatting_avatar.yaml;configs/instant_avatar.yaml --dat_dir <path/to/subject> --deform_on 1 --model_path <path/to/subject/output-splatting/> 

Evaluation
python eval_animate.py --config "configs/splatting_avatar.yaml;configs/instant_avatar.yaml" --dat_dir /path-to/female-3-casual --pc_dir /path-to/female-3-casual/output-splatting/last_checkpoint/point_cloud/iteration_50000 --anim_fn /path-to/aist_demo.npz

