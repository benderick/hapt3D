import click
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
import yaml
import datasets.dataloader as dataloader
from models.hapt3d import HAPT3D as HAPT3D
from pytorch_lightning import loggers as pl_loggers

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)

def main(config, weights):
    assert weights is not None
    # Loading cfg from checkpoint if available
    ckpt = torch.load(weights)
    if 'hyper_parameters' in ckpt.keys():
        cfg = ckpt['hyper_parameters']
    else:  
        cfg = yaml.safe_load(open(config))
    cfg['data_path'] = '/home/matteo/Code/HAPT3D/dataset/'
    cfg['test'] = {}
    cfg['test']['dump_metrics'] = True
    # Load data and model
    data = dataloader.StatDataModule(cfg)
    
    cfg['val']['min_n_points_fruit'] = 30
    cfg['val']['min_n_points_trunk'] = 576
    cfg['val']['min_n_points_tree'] = 1614

    # #### TO BE REMOVED ####
    # ''' Next line is only for extracting validation results --> then remove'''
    # data.data_test = data.data_val
    # ######################

    model = HAPT3D.load_from_checkpoint(weights,cfg=cfg,viz=False)

    tb_logger = pl_loggers.TensorBoardLogger('evaluations/'+cfg['experiment']['id'],
                                             default_hp_metric=False)
    # Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      max_epochs= cfg['train']['max_epoch'],
                      num_sanity_val_steps=0)
    model.min_n_points_fruit = 54
    trainer.test(model, data)

if __name__ == "__main__":
    main()
