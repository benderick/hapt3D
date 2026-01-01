#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_v2.py - 训练脚本

功能:
- 支持HFE/CDAG/HCL模块的配置系统
- 配置验证和摘要显示
- 增强的日志记录

用法:
    # 指定配置文件训练
    python train_v2.py -c config/config_ours.yaml
    
    # 覆盖配置项
    python train_v2.py -c config/config_ours.yaml --lr 0.001 --epochs 200
    
    # 调试模式
    python train_v2.py -c config/config_ours.yaml --debug

注意: 运行消融实验请使用 run_ablation.py
"""

import click
import os
import sys
from os.path import join, dirname, abspath
from datetime import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml

import datasets.dataloader as dataloader
from models.hapt3d_ours import HAPT3D
from utils.func import EarlyStoppingWithWarmup
from utils.config_manager import ConfigManager, load_config


def setup_callbacks(cfg: ConfigManager, tb_logger):
    """
    设置训练回调函数
    
    Args:
        cfg: 配置管理器
        tb_logger: TensorBoard日志器
    
    Returns:
        回调函数列表
    """
    callbacks = []
    
    # === 模型检查点保存 ===
    
    # 按mIoU保存
    checkpoint_saver_miou = ModelCheckpoint(
        monitor='Metrics_ious/miou',
        filename='best-miou-epoch-{epoch:02d}',
        auto_insert_metric_name=False,
        mode='max',
        verbose=False,
        save_last=True
    )
    callbacks.append(checkpoint_saver_miou)
    
    # 按mPQ保存
    checkpoint_saver_pq = ModelCheckpoint(
        monitor='Metrics_pqs/mpq',
        filename='best-mpq-epoch-{epoch:02d}',
        auto_insert_metric_name=False,
        mode='max',
        verbose=False,
        save_last=False
    )
    callbacks.append(checkpoint_saver_pq)
    
    # 按PQ_tree保存
    checkpoint_saver_pqh = ModelCheckpoint(
        monitor='Metrics_pqs/pq_h',
        filename='best-pqh-epoch-{epoch:02d}',
        auto_insert_metric_name=False,
        mode='max',
        verbose=False,
        save_last=False
    )
    callbacks.append(checkpoint_saver_pqh)
    
    # 按实例损失保存
    checkpoint_saver_ins1loss = ModelCheckpoint(
        monitor='Loss/ins1_loss_val',
        filename='best-ins1-epoch-{epoch:02d}',
        auto_insert_metric_name=False,
        mode='min',
        verbose=False,
        save_last=False
    )
    callbacks.append(checkpoint_saver_ins1loss)
    
    checkpoint_saver_ins2loss = ModelCheckpoint(
        monitor='Loss/ins2_loss_val',
        filename='best-ins2-epoch-{epoch:02d}',
        auto_insert_metric_name=False,
        mode='min',
        verbose=False,
        save_last=False
    )
    callbacks.append(checkpoint_saver_ins2loss)
    
    # === 早停 ===
    early_stop_cfg = cfg.get('train.early_stopping', {})
    if early_stop_cfg.get('enabled', True):
        early_stopping = EarlyStoppingWithWarmup(
            monitor=early_stop_cfg.get('monitor', 'Metrics_pqs/mpq'),
            mode='max',
            warmup=early_stop_cfg.get('warmup', 100),
            patience=early_stop_cfg.get('patience', 50),
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # === 学习率监控 ===
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def print_experiment_header(cfg: ConfigManager):
    """打印实验信息头"""
    print("\n" + "=" * 70)
    print("HAPT3D 训练启动")
    print("=" * 70)
    print(f"实验ID: {cfg.experiment_id}")
    print(f"描述: {cfg.get('experiment.description', 'N/A')}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print("模块配置:")
    print(f"  HFE:  {'✓ 启用' if cfg.hfe_enabled else '✗ 禁用'}")
    print(f"  CDAG: {'✓ 启用' if cfg.cdag_enabled else '✗ 禁用'}")
    print(f"  HCL:  {'✓ 启用 (λ={cfg.hcl_weight})' if cfg.hcl_enabled else '✗ 禁用'}")
    print("-" * 70)
    print("训练参数:")
    print(f"  学习率: {cfg.learning_rate}")
    print(f"  最大轮数: {cfg.max_epochs}")
    print(f"  批大小: {cfg.batch_size}")
    print(f"  体素分辨率: {cfg.voxel_resolution}")
    print("=" * 70 + "\n")


@click.command()
@click.option('--config', '-c', type=str, required=True,
              help='配置文件路径 (.yaml)')
@click.option('--weights', '-w', type=str, default=None,
              help='预训练权重路径 (.ckpt)，仅加载权重不恢复训练')
@click.option('--checkpoint', '-ckpt', type=str, default=None,
              help='检查点路径 (.ckpt)，用于恢复训练')
@click.option('--lr', type=float, default=None,
              help='覆盖配置文件中的学习率')
@click.option('--epochs', type=int, default=None,
              help='覆盖配置文件中的最大训练轮数')
@click.option('--batch-size', type=int, default=None,
              help='覆盖配置文件中的批大小')
@click.option('--gpus', type=int, default=None,
              help='覆盖配置文件中的GPU数量')
@click.option('--seed', type=int, default=None,
              help='随机种子')
@click.option('--debug', is_flag=True, default=False,
              help='调试模式，减少数据量快速验证')
def main(config, weights, checkpoint, lr, epochs, batch_size, gpus, seed, debug):
    """
    HAPT3D 训练脚本
    
    支持通过配置文件灵活切换HFE、CDAG、HCL模块。
    运行消融实验请使用 run_ablation.py
    """
    
    # === 加载配置 ===
    overrides = {}
    if lr is not None:
        overrides['train'] = overrides.get('train', {})
        overrides['train']['lr'] = lr
    if epochs is not None:
        overrides['train'] = overrides.get('train', {})
        overrides['train']['max_epoch'] = epochs
    if batch_size is not None:
        overrides['train'] = overrides.get('train', {})
        overrides['train']['batch_size'] = batch_size
    if gpus is not None:
        overrides['train'] = overrides.get('train', {})
        overrides['train']['n_gpus'] = gpus
    
    cfg = load_config(config, **overrides) if overrides else load_config(config)
    
    # 调试模式特殊处理
    if debug:
        cfg.set('train.max_epoch', 2)
        cfg.set('train.overfit', True)
        print("[DEBUG] 调试模式启用，最大轮数设为2")
    
    # === 配置验证 ===
    if not cfg.validate():
        print("[错误] 配置验证失败，请检查配置文件")
        sys.exit(1)
    
    # === 设置随机种子 ===
    actual_seed = seed if seed is not None else cfg.get('experiment.seed', 42)
    seed_everything(actual_seed, workers=True)
    print(f"[INFO] 随机种子: {actual_seed}")
    
    # === 打印实验信息 ===
    print_experiment_header(cfg)
    
    # === 从检查点恢复配置 ===
    raw_cfg = cfg.config  # 获取原始字典用于兼容现有代码
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        if 'hyper_parameters' in ckpt.keys():
            print("[INFO] 从检查点加载超参数配置")
            raw_cfg = ckpt['hyper_parameters']
            raw_cfg['data_path'] = cfg.get('data.path', 'data/hopt3d')
    
    # === 数据加载 ===
    print("[INFO] 加载数据...")
    data = dataloader.StatDataModule(raw_cfg)
    
    # === 模型初始化 ===
    print("[INFO] 初始化模型...")
    if weights is None:
        model = HAPT3D(raw_cfg)
    else:
        model = HAPT3D.load_from_checkpoint(weights, cfg=raw_cfg, viz=False)
        print(f"[INFO] 从预训练权重加载: {weights}")
    
    # === 设置日志器 ===
    experiment_name = cfg.experiment_id
    if debug:
        experiment_name = f"debug_{experiment_name}"
    
    tb_logger = pl_loggers.TensorBoardLogger(
        f'experiments/{experiment_name}',
        default_hp_metric=False
    )
    
    # 保存当前配置到实验目录
    config_save_path = os.path.join(tb_logger.log_dir, 'config.yaml')
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    cfg.save(config_save_path)
    print(f"[INFO] 配置已保存至: {config_save_path}")
    
    # === 设置回调 ===
    callbacks = setup_callbacks(cfg, tb_logger)
    
    # === 设置训练器 ===
    trainer = Trainer(
        accelerator='gpu',
        devices=cfg.get('train.n_gpus', 1),
        logger=tb_logger,
        resume_from_checkpoint=checkpoint,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=15,
        num_sanity_val_steps=1 if not debug else 0,
        callbacks=callbacks,
        deterministic=True  # 确保可重复性
    )
    
    torch.set_float32_matmul_precision('medium')
    
    # === 开始训练 ===
    print("[INFO] 开始训练...")
    trainer.fit(model, data)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳模型保存于: {tb_logger.log_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
