# RMAvatar
Getting Started
Create conda env with pytorch.
conda create -n splatting python=3.10
conda activate rmavatar

# pytorch 2.0.1+cu117 is tested

# install other dependencies
cd submodules/diff-gaussian-rasterization
pip install .
cd ../submodules/simple-knn
pip install .
pip install -r requirements.txt

Training
python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir <path/to/subject>
Evaluation
python eval_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir <path/to/model_path>
