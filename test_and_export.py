#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_ply.py - 导出模型预测结果为PLY点云文件

功能:
- 加载模型并对测试集进行推理
- 导出原始点云、Ground Truth、预测结果为PLY格式
- 支持CloudCompare等工具直接查看

数据格式说明:
    HOPS数据集返回的batch字典包含以下键:
    - positions   : (N, 3) Float32 - 点云坐标 (主键)
    - colors      : (N, 3) Float32 - RGB颜色值 [0-255]
    - semantic    : (N, 1) Float64 - 语义标签 [0-5]
    - semantic_h  : (N, 1) Float64 - 层次化语义标签 [0-1]
    - instance    : (N, 1) Float64 - 实例标签
    - instance_h  : (N, 1) Float64 - 树木实例标签
    
    模型预测输出:
    - output_sem  : TensorField - 语义分割logits (N, 6)
    - offsets1    : TensorField - 实例偏移向量 (N, D)
    - offsets2    : TensorField - 树木偏移向量 (N, D)
    
    使用 TensorField.features_at(batch_id) 获取具体特征

用法:
    # 导出单个样本
    python export_ply.py -w checkpoints/best-mpq.ckpt -i 0
    
    # 导出多个样本
    python export_ply.py -w checkpoints/best-mpq.ckpt -i 0 1 2 3 4
    
    # 指定输出目录
    python export_ply.py -w checkpoints/best-mpq.ckpt -i 0 -o ply_results/
    
    # 批量导出所有测试集
    python export_ply.py -w checkpoints/best-mpq.ckpt --all -n 10
"""

import click
import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.hapt3d_ours import HAPT3D
from datasets.dataset import HAPT3DDataset
from utils.func import TensorField
import MinkowskiEngine as ME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')


# ============================================================================
# PLY导出函数
# ============================================================================

def save_ply_with_all_fields(data_dict, output_path):
    """
    保存点云为PLY格式，包含所有字段作为vertex properties
    
    Args:
        data_dict: 数据字典，包含以下键:
            - positions   : (N, 3) 点云坐标
            - colors      : (N, 3) RGB颜色
            - semantic    : (N,) 或 (N, 1) 语义标签
            - semantic_h  : (N,) 或 (N, 1) 层次化语义标签
            - instance    : (N,) 或 (N, 1) 实例标签
            - instance_h  : (N,) 或 (N, 1) 树木实例标签
        output_path: 输出PLY文件路径
    """
    from plyfile import PlyData, PlyElement
    
    # 提取数据
    positions = data_dict['positions']
    colors = data_dict['colors']
    semantic = data_dict['semantic'].squeeze() if data_dict['semantic'].ndim > 1 else data_dict['semantic']
    semantic_h = data_dict['semantic_h'].squeeze() if data_dict['semantic_h'].ndim > 1 else data_dict['semantic_h']
    instance = data_dict['instance'].squeeze() if data_dict['instance'].ndim > 1 else data_dict['instance']
    instance_h = data_dict['instance_h'].squeeze() if data_dict['instance_h'].ndim > 1 else data_dict['instance_h']
    
    # 归一化颜色到[0, 255]
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    # 构造vertex数据（使用numpy structured array）
    n_points = len(positions)
    vertex_data = np.zeros(
        n_points,
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),           # positions
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),  # colors
            ('semantic', 'i4'),                               # semantic
            ('semantic_h', 'i4'),                             # semantic_h
            ('instance', 'i4'),                               # instance
            ('instance_h', 'i4'),                             # instance_h
        ]
    )
    
    # 填充数据
    vertex_data['x'] = positions[:, 0]
    vertex_data['y'] = positions[:, 1]
    vertex_data['z'] = positions[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    vertex_data['semantic'] = semantic.astype(np.int32)
    vertex_data['semantic_h'] = semantic_h.astype(np.int32)
    vertex_data['instance'] = instance.astype(np.int32)
    vertex_data['instance_h'] = instance_h.astype(np.int32)
    
    # 创建PLY元素
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # 保存PLY文件
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(str(output_path))
    
    print(f"  ✓ {output_path.name} (含6个标量场)")


def colorize_semantic(labels):
    """
    根据语义标签生成颜色
    
    Args:
        labels: [N] 语义标签
    
    Returns:
        colors: [N, 3] RGB颜色
    """
    SEMANTIC_COLORS = {
        0: [0.0, 0.0, 0.0],        # void
        1: [0.82, 0.71, 0.55],     # ground
        2: [0.18, 0.55, 0.34],     # plant
        3: [1.0, 0.27, 0.0],       # fruit
        4: [0.55, 0.27, 0.07],     # trunk
        5: [0.44, 0.5, 0.56],      # pole
    }
    
    colors = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        colors[i] = SEMANTIC_COLORS.get(int(label), [0.5, 0.5, 0.5])
    
    return colors


def colorize_instance(instances):
    """
    根据实例ID生成独特颜色
    
    Args:
        instances: [N] 实例ID
    
    Returns:
        colors: [N, 3] RGB颜色
    """
    import distinctipy
    
    unique_instances, inverse, counts = np.unique(
        instances, return_inverse=True, return_counts=True
    )
    
    n_instances = len(unique_instances)
    colors_list = distinctipy.get_colors(
        n_instances,
        exclude_colors=[(c, c, c) for c in np.arange(0, 1.01, 0.01)]
    )
    
    # 最大实例（背景）用灰色
    bg_idx = np.argmax(counts)
    colors_list[bg_idx] = (0.7, 0.7, 0.7)
    
    colors = np.array([colors_list[idx] for idx in inverse])
    
    return colors


# ============================================================================
# 模型推理
# ============================================================================

def model_inference(model, sample, voxel_resolution, device):
    """
    模型推理，返回与Ground Truth格式一致的预测字典
    
    Args:
        model: HAPT3D模型
        sample: 数据集样本（GT格式）
        voxel_resolution: 体素分辨率
        device: 设备
    
    Returns:
        predictions: 预测字典，格式与GT一致
            - positions   : (N, 3) Float32
            - colors      : (N, 3) Float32
            - semantic    : (N, 1) Float64
            - semantic_h  : (N, 1) Float64
            - instance    : (N, 1) Float64
            - instance_h  : (N, 1) Float64
    """
    model.eval()
    
    # ==================== 数据准备 ====================
    if 'positions' in sample:
        points_data = sample['positions']
    elif 'points' in sample:
        points_data = sample['points']
    else:
        raise KeyError("数据中既没有 'positions' 也没有 'points' 键")
    
    colors_data = sample['colors']
    
    # 转换为tensor并移到GPU
    if isinstance(points_data, np.ndarray):
        points_data = torch.from_numpy(points_data).float()
    if isinstance(colors_data, np.ndarray):
        colors_data = torch.from_numpy(colors_data).float()
    
    # TensorField内部使用numpy，需要保持在CPU上创建
    points_data = points_data.float()
    colors_data = colors_data.float()
    
    # ==================== 创建TensorField ====================
    tensorfield = {
        "points": [points_data],
        "feats": [colors_data]
    }
    dense_input = TensorField(tensorfield, voxel_resolution=voxel_resolution)
    
    # ==================== 模型前向传播 ====================
    # 模型的forward方法会自动处理设备转换（通过.sparse()和model内部）
    with torch.no_grad():
        output_sem, offsets1, offsets2 = model(dense_input)
    
    # ==================== 提取预测结果 ====================
    # 语义预测 - (N, n_classes) -> (N,)
    sem_pred_logits = output_sem.features_at(0)  # (N, 6)
    sem_pred = torch.argmax(sem_pred_logits, dim=1)  # (N,)
    
    # 层次化语义预测 - 0: background, 1: tree (fruit + trunk)
    sem_h_pred = torch.zeros_like(sem_pred)
    sem_h_pred[torch.logical_or(sem_pred == 3, sem_pred == 4)] = 1  # fruit=3, trunk=4
    
    # 偏移向量
    offsets_ins = offsets1.features_at(0)  # (N, D)
    offsets_tree = offsets2.features_at(0)  # (N, D)
    offset_ins_xyz = offsets_ins[:, :3] if offsets_ins.shape[1] >= 3 else offsets_ins
    offset_tree_xyz = offsets_tree[:, :3] if offsets_tree.shape[1] >= 3 else offsets_tree
    
    # ==================== HDBSCAN聚类得到实例预测 ====================
    from hdbscan import HDBSCAN
    
    coords = dense_input.coordinates_at(0) * voxel_resolution  # 恢复原始坐标
    
    # 实例聚类（fruit + trunk）
    centers_ins = (coords + offset_ins_xyz).cpu().numpy()
    clusterer_ins = HDBSCAN(min_cluster_size=50, min_samples=10)
    ins_pred = clusterer_ins.fit_predict(centers_ins)
    ins_pred = torch.from_numpy(ins_pred).long().to(device)
    
    # 树木聚类（hierarchy level）
    centers_tree = (coords + offset_tree_xyz).cpu().numpy()
    clusterer_tree = HDBSCAN(min_cluster_size=200, min_samples=20)
    ins_h_pred = clusterer_tree.fit_predict(centers_tree)
    ins_h_pred = torch.from_numpy(ins_h_pred).long().to(device)
    
    # ==================== 整理成GT格式的字典 ====================
    predictions = {
        # 坐标和颜色（与GT相同）
        'positions': points_data.cpu().float(),           # (N, 3) Float32
        'colors': colors_data.cpu().float(),              # (N, 3) Float32
        
        # 语义预测（转换为Float64以匹配GT格式）
        'semantic': sem_pred.cpu().unsqueeze(1).double(), # (N, 1) Float64
        'semantic_h': sem_h_pred.cpu().unsqueeze(1).double(),  # (N, 1) Float64
        
        # 实例预测（转换为Float64以匹配GT格式）
        'instance': ins_pred.cpu().unsqueeze(1).double(), # (N, 1) Float64
        'instance_h': ins_h_pred.cpu().unsqueeze(1).double(),  # (N, 1) Float64
        
        # 额外信息（用于调试）
        '_offsets_instance': offsets_ins.cpu(),           # (N, D) 实例偏移向量
        '_offsets_tree': offsets_tree.cpu(),              # (N, D) 树木偏移向量
        '_logits': sem_pred_logits.cpu(),                 # (N, 6) 语义logits
    }
    
    return predictions


def inference_and_export(model, sample, voxel_resolution, output_dir, sample_idx):
    """
    对单个样本进行推理并导出PLY文件
    
    Args:
        model: HAPT3D模型
        sample: 数据集样本（Ground Truth）
        voxel_resolution: 体素分辨率
        output_dir: 输出目录
        sample_idx: 样本索引
    """
    device = next(model.parameters()).device
    
    # ==================== 模型推理 ====================
    print(f"\n样本 {sample_idx}:")
    print(f"  推理中...")
    
    # 获取与GT格式一致的预测结果
    predictions = model_inference(model, sample, voxel_resolution, device)
    
    # ==================== 打印格式对比 ====================
    print(f"\n  Ground Truth 格式:")
    coords_key = 'positions' if 'positions' in sample else 'points'
    for key in [coords_key, 'colors', 'semantic', 'semantic_h', 'instance', 'instance_h']:
        if key in sample:
            value = sample[key]
            if torch.is_tensor(value):
                print(f"    - {key:12s}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    
    print(f"\n  模型预测格式:")
    for key in ['positions', 'colors', 'semantic', 'semantic_h', 'instance', 'instance_h']:
        if key in predictions:
            value = predictions[key]
            print(f"    - {key:12s}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    
    # ==================== 提取数据用于导出 ====================
    points = predictions['positions'].numpy()
    colors = predictions['colors'].numpy()
    
    # Ground Truth
    sem_gt = sample['semantic'].squeeze().numpy().astype(np.int32)
    ins_gt = sample['instance'].squeeze().numpy().astype(np.int32) if 'instance' in sample else None
    ins_h_gt = sample['instance_h'].squeeze().numpy().astype(np.int32) if 'instance_h' in sample else None
    
    # 预测结果
    sem_pred = predictions['semantic'].squeeze().numpy().astype(np.int32)
    sem_h_pred = predictions['semantic_h'].squeeze().numpy().astype(np.int32)
    ins_pred = predictions['instance'].squeeze().numpy().astype(np.int32)
    ins_h_pred = predictions['instance_h'].squeeze().numpy().astype(np.int32)
    
    # ==================== 导出PLY文件 ====================
    
    # ==================== 导出PLY文件 ====================
    # 创建输出目录
    sample_dir = output_dir / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  点数: {len(points)}")
    print(f"  输出目录: {sample_dir}")
    
    # 准备Ground Truth字典
    gt_dict = {
        'positions': points,
        'colors': colors,
        'semantic': sem_gt,
        'semantic_h': sample['semantic_h'].squeeze().numpy().astype(np.int32),
        'instance': ins_gt if ins_gt is not None else np.zeros_like(sem_gt),
        'instance_h': ins_h_gt if ins_h_gt is not None else np.zeros_like(sem_gt),
    }
    
    # 准备预测字典
    pred_dict = {
        'positions': points,
        'colors': colors,
        'semantic': sem_pred,
        'semantic_h': sem_h_pred,
        'instance': ins_pred,
        'instance_h': ins_h_pred,
    }
    
    # 保存Ground Truth PLY（包含所有字段）
    save_ply_with_all_fields(
        gt_dict,
        sample_dir / "ground_truth.ply"
    )
    
    # 保存预测结果PLY（包含所有字段）
    save_ply_with_all_fields(
        pred_dict,
        sample_dir / "predictions.ply"
    )
    
    # ==================== 保存额外调试信息（NPZ格式）====================
    # 保存偏移向量和logits用于调试
    debug_path = sample_dir / "debug_info.npz"
    np.savez(
        debug_path,
        offsets_instance=predictions['_offsets_instance'].numpy(),
        offsets_tree=predictions['_offsets_tree'].numpy(),
        logits=predictions['_logits'].numpy(),
    )
    print(f"  ✓ 调试信息已保存: debug_info.npz")
    
    # ==================== 计算统计信息 ====================
    
    # ==================== 计算统计信息 ====================
    stats = {
        'sample_idx': sample_idx,
        'num_points': len(points),
        'semantic_accuracy': np.mean(sem_pred == sem_gt) * 100,
        'num_instances_pred': len(np.unique(ins_pred)),
        'num_trees_pred': len(np.unique(ins_h_pred)),
    }
    
    if ins_gt is not None:
        stats['num_instances_gt'] = len(np.unique(ins_gt))
    if ins_h_gt is not None:
        stats['num_trees_gt'] = len(np.unique(ins_h_gt))
    
    # 保存统计信息
    stats_path = sample_dir / "stats.txt"
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    return stats


# ============================================================================
# 主函数
# ============================================================================

@click.command()
@click.option('--weights', '-w', type=str, required=True,
              help='模型权重路径 (.ckpt)')
@click.option('--config', '-c', type=str, default=None,
              help='配置文件路径 (.yaml)')
@click.option('--data_path', '-d', type=str, default=None,
              help='数据集路径')
@click.option('--indices', '-i', multiple=True, type=int,
              help='要导出的样本索引（可指定多个）')
@click.option('--split', type=str, default='val',
              help='数据集划分: train/val/test')
@click.option('--output', '-o', type=str, default='ply_exports/',
              help='输出目录')
@click.option('--all', 'export_all', is_flag=True, default=False,
              help='导出所有样本')
@click.option('--num_samples', '-n', type=int, default=None,
              help='导出样本数量（与--all一起使用）')
def main(weights, config, data_path, indices, split, output, export_all, num_samples):
    """
    HAPT3D 模型预测结果PLY导出工具
    
    导出的文件可以在CloudCompare中打开查看
    """
    print("\n" + "="*70)
    print("HAPT3D PLY点云导出工具")
    print("="*70 + "\n")
    
    # 加载配置
    print("📋 加载配置...")
    ckpt = torch.load(weights, map_location='cpu')
    if 'hyper_parameters' in ckpt:
        cfg = ckpt['hyper_parameters']
    elif config:
        cfg = yaml.safe_load(open(config))
    else:
        raise ValueError("无法从checkpoint加载配置，请使用 --config 指定配置文件")
    
    if data_path:
        if 'data' in cfg:
            cfg['data']['path'] = data_path
        else:
            cfg['data_path'] = data_path
    
    print(f"  实验ID: {cfg['experiment']['id']}")
    print(f"  体素分辨率: {cfg['train']['voxel_resolution']}")
    
    # 加载模型
    print("\n🤖 加载模型...")
    model = HAPT3D.load_from_checkpoint(weights, cfg=cfg, viz=False)
    model = model.to(device)
    model.eval()
    print(f"  模型已加载到: {device}")
    
    # 加载数据集
    print(f"\n📦 加载数据集 ({split})...")
    data_path_value = cfg.get('data', {}).get('path', cfg.get('data_path', 'data/hopt3d'))
    dataset = HAPT3DDataset(
        data_path=data_path_value,
        config=cfg,
        split=split,
        overfit=False
    )
    print(f"  数据集大小: {len(dataset)}")
    
    # 确定要导出的样本
    if export_all:
        if num_samples:
            indices = range(min(num_samples, len(dataset)))
        else:
            indices = range(len(dataset))
    elif not indices:
        indices = [0]  # 默认导出第一个样本
    
    # 创建输出目录
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 输出目录: {output_dir}")
    
    # 批量导出
    print(f"\n🎨 开始导出 (共 {len(indices)} 个样本)...\n")
    
    voxel_resolution = cfg['train']['voxel_resolution']
    all_stats = []
    
    for idx in tqdm(indices, desc="导出PLY"):
        if idx >= len(dataset):
            print(f"⚠️  警告: 索引 {idx} 超出数据集范围，跳过")
            continue
        
        sample = dataset[idx]
        
        try:
            stats = inference_and_export(
                model, sample, voxel_resolution, output_dir, idx
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"❌ 样本 {idx} 导出失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存总体统计信息
    if all_stats:
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("HAPT3D PLY导出总结\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"导出样本数: {len(all_stats)}\n")
            f.write(f"数据集划分: {split}\n")
            f.write(f"模型权重: {weights}\n\n")
            
            avg_acc = np.mean([s['semantic_accuracy'] for s in all_stats])
            f.write(f"平均语义准确率: {avg_acc:.2f}%\n\n")
            
            f.write("各样本详情:\n")
            f.write("-" * 70 + "\n")
            for stats in all_stats:
                f.write(f"样本 {stats['sample_idx']:3d}: "
                       f"点数={stats['num_points']:6d}, "
                       f"准确率={stats['semantic_accuracy']:.2f}%\n")
        
        print(f"\n✓ 总结保存至: {summary_path}")
    
    print("\n" + "="*70)
    print("✅ 导出完成！")
    print("="*70)
    print(f"\n📁 输出目录: {output_dir}")
    print("\n💡 使用CloudCompare打开PLY文件:")
    print(f"   cloudcompare.CloudCompare {output_dir}/sample_000/ground_truth.ply")
    print(f"   cloudcompare.CloudCompare {output_dir}/sample_000/predictions.ply")
    print("\n文件说明:")
    print("  ground_truth.ply  - Ground Truth（包含所有标量场）")
    print("  predictions.ply   - 模型预测（包含所有标量场）")
    print("  debug_info.npz    - 调试信息（偏移向量、logits）")
    print("\n标量场（Scalar Fields）:")
    print("  - positions   : (x, y, z) 点云坐标")
    print("  - colors      : (red, green, blue) RGB颜色")
    print("  - semantic    : 语义标签 [0-5]")
    print("  - semantic_h  : 层次化语义标签 [0-1]")
    print("  - instance    : 实例标签")
    print("  - instance_h  : 树木实例标签")
    print()


if __name__ == "__main__":
    main()
