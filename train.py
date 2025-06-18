import click
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import datasets.dataloader as dataloader
from models.hapt3d import HAPT3D as HAPT3D
from utils.func import EarlyStoppingWithWarmup



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
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)

def main(config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    # Loading cfg from checkpoint if available
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        if 'hyper_parameters' in ckpt.keys():
            cfg = ckpt['hyper_parameters']
            cfg['data_path'] = '/home/matteo/Code/HAPT3D/dataset'


    # Load data and model
    data = dataloader.StatDataModule(cfg)
    
    if weights is None:
        model = HAPT3D(cfg)
    else:
        model = HAPT3D.load_from_checkpoint(weights,cfg=cfg,viz=False)
       
    # Add callbacks:
    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    checkpoint_saver_miou = ModelCheckpoint(monitor='Metrics_ious/miou',
                                 filename='best-miou-epoch-{epoch:02d}',
                                 auto_insert_metric_name=False,
                                 mode='max',
                                 verbose=False,
                                 save_last=True)
    
    checkpoint_saver_pq = ModelCheckpoint(monitor='Metrics_pqs/mpq',
                                 filename='best-mpq-epoch-{epoch:02d}',
                                 auto_insert_metric_name=False,
                                 mode='max',
                                 verbose=False,
                                 save_last=False)

    checkpoint_saver_pqh = ModelCheckpoint(monitor='Metrics_pqs/pq_h',
                                 filename='best-pqh-epoch-{epoch:02d}',
                                 auto_insert_metric_name=False,
                                 mode='max',
                                 verbose=False,
                                 save_last=False)
    
    checkpoint_saver_ins1loss = ModelCheckpoint(monitor='Loss/ins1_loss_val',
                                 filename='best-ins1-epoch-{epoch:02d}',
                                 auto_insert_metric_name=False,
                                 mode='min',
                                 verbose=False,
                                 save_last=False)
    
    checkpoint_saver_ins2loss = ModelCheckpoint(monitor='Loss/ins2_loss_val',
                                 filename='best-ins2-epoch-{epoch:02d}',
                                 auto_insert_metric_name=False,
                                 mode='min',
                                 verbose=False,
                                 save_last=False)

    early_stopping = EarlyStoppingWithWarmup(monitor='Metrics_pqs/mpq', mode='max', warmup=100, patience=50, verbose=True)
    
    # Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs= cfg['train']['max_epoch'],
                      log_every_n_steps=15,
                      num_sanity_val_steps=1,
                      callbacks=[checkpoint_saver_miou, checkpoint_saver_pq, checkpoint_saver_pqh, checkpoint_saver_ins1loss, checkpoint_saver_ins2loss])
    # Train
    trainer.fit(model, data)

    # Test
    # trainer.test(model, data)

if __name__ == "__main__":
    main()