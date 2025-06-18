# INSTALL

conda create --name hapt3d python=3.9
conda activate hapt3d
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
pip install numpy==1.24.2
pip install setuptools==60.0
pip install pykeops --no-cache-dir
(pip install ninja)
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
pip install pytorch-lightning==1.9.0 --no-deps
pip install fsspec
pip install lightning-utilities
pip install tqdm
pip install pyyaml
pip install torchmetrics==1.4.1
pip install ipdb
pip install open3d
pip install tensorboard
pip install hdbscan
pip install distinctipy
pip install optuna==3.6.1
pip install optuna-integration
