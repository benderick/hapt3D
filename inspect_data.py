import click
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
import yaml
import datasets.dataloader as dataloader
from models.hapt3d_ours import HAPT3D as HAPT3D
from pytorch_lightning import loggers as pl_loggers
torch.set_float32_matmul_precision('medium')

cfg = yaml.safe_load(open('config/config_baseline.yaml'))
data = dataloader.StatDataModule(cfg)
data.setup('test')
for i in range(len(data.data_test)):
    data.data_test[i]
    print("*"*20)
