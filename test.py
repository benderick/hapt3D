import click
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
import yaml
import datasets.dataloader as dataloader
from models.hapt3d_ours import HAPT3D as HAPT3D
from pytorch_lightning import loggers as pl_loggers
torch.set_float32_matmul_precision('medium')

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default='config/config_baseline.yaml')
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default='experiments/baseline/lightning_logs/version_0/checkpoints/best-miou-epoch-06.ckpt')
@click.option('--data_path',
              '-d',
              type=str,
              help='path to the dataset directory',
              default=None)

def main(config, weights, data_path):
    assert weights is not None, "请使用 --weights 指定模型权重路径"
    
    # 从checkpoint加载配置（如果有）
    ckpt = torch.load(weights)
    if 'hyper_parameters' in ckpt.keys():
        cfg = ckpt['hyper_parameters']
    else:  
        cfg = yaml.safe_load(open(config))
    
    # 设置数据路径（命令行参数优先，否则使用配置文件中的路径）
    if data_path is not None:
        cfg['data_path'] = data_path

    # 加载数据模块
    data = dataloader.StatDataModule(cfg)
    
    # 设置验证参数（最小点数阈值）
    cfg['val']['min_n_points_fruit'] = 54   # 果实最小点数
    cfg['val']['min_n_points_trunk'] = 576  # 树干最小点数
    cfg['val']['min_n_points_tree'] = 1614  # 树木最小点数

    # 加载模型
    model = HAPT3D.load_from_checkpoint(weights, cfg=cfg, viz=True)

    # 设置日志
    tb_logger = pl_loggers.TensorBoardLogger('evaluations/'+cfg['experiment']['id'],
                                             default_hp_metric=False)
    # 设置Trainer
    trainer = Trainer(
        devices=cfg['train']['n_gpus'],
        accelerator='gpu',
        logger=tb_logger,
        max_epochs=cfg['train']['max_epoch'],
        num_sanity_val_steps=0
    )
    
    # 运行测试
    trainer.test(model, data)

if __name__ == "__main__":
    main()
