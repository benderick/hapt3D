# HAPT3D

### Train
Run `python train.py --config config/config_full.yaml`. Remember to change the path to the dataset folder in the config file and in the `train.py` file.

### Testing
Run `python test.py -w <file>`. Remember to change the path to the dataset folder in the config file and in the `test.py` file. If you want to test on the validation set, uncomment lines 41-44 in `test.py`.

### Installation
After struggling a bit to install MinkowskiEngine, the procedure below is the one that worked out on my machine (operations to be done in that specific order):

```
    conda create --name hapt3d python=3.9
    conda activate hapt3d
    pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
    pip install numpy==1.24.2
    pip install setuptools==60.0
    pip install pykeops --no-cache-di
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
    pip install torchmetrics
    pip install hdbscan
    pip install distinctipy
    pip install optuna==3.6.1
    pip install optuna-integration
```

Good luck :)

### Docker
Alternatively, you could simply use docker. Build it first via `make build`, then you can train via doing `make train` and test with `make test CHECKPOINT=<file>`.

