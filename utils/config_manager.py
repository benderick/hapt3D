# =============================================================================
# 配置管理器
# 描述: 提供配置文件的加载、验证和便捷访问功能
# =============================================================================

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class ConfigManager:
    """
    配置管理器类
    
    功能:
    - 从YAML文件加载配置
    - 提供默认值
    - 配置验证
    - 便捷的配置访问方式
    """
    
    # 默认配置模板
    DEFAULT_CONFIG = {
        'experiment': {
            'id': 'default',
            'description': '',
            'seed': 42
        },
        'data': {
            'path': 'data/hopt3d',
            'train_split': 'train',
            'val_split': 'val',
            'test_split': 'test'
        },
        'network': {
            'backbone': 'MinkUNet14A',
            'in_channels': 3,
            'out_channels': 256,
            'tanh': True,
            'embeddings_only': False,
            'skip': 'full',
            'hfe': {'enabled': False},
            'cdag': {'enabled': False}
        },
        'tasks': {
            'semantic_segmentation': {
                'enabled': True,
                'n_classes': 6,
                'ignore_idx': 0,
                'stuff_ids': [1, 2, 5],
                'things_ids': [3, 4]
            },
            'instance_segmentation': {
                'enabled': True,
                'variance': 0.01,
                'fg_classes': [3, 4]
            },
            'tree_segmentation': {
                'enabled': True,
                'variance': 0.5
            }
        },
        'loss': {
            'semantic': {'type': 'cross_entropy', 'weight': 1.0},
            'instance': {'type': 'lovasz', 'weight': 1.0},
            'tree': {'type': 'lovasz', 'weight': 1.0},
            'hcl': {'enabled': False, 'weight': 0.0}
        },
        'train': {
            'max_epoch': 500,
            'lr': 0.005,
            'weight_decay': 0.01,
            'batch_size': 1,
            'n_gpus': 1,
            'workers': 4,
            'voxel_resolution': 0.003,
            'overfit': False
        },
        'val': {
            'min_n_points_fruit': 60,
            'min_n_points_trunk': 250,
            'min_n_points_tree': 1000,
            'pq_from_epoch': 50
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典（优先级高于配置文件）
        """
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        
        if config_path:
            self._load_from_file(config_path)
        
        if config_dict:
            self._deep_update(self._config, config_dict)
    
    def _deep_copy(self, d: Dict) -> Dict:
        """深拷贝字典"""
        import copy
        return copy.deepcopy(d)
    
    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        """
        递归更新字典
        
        Args:
            base: 基础字典
            update: 更新字典
        
        Returns:
            更新后的字典
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    def _load_from_file(self, config_path: str):
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
        
        if file_config:
            self._deep_update(self._config, file_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持点分隔的路径）
        
        Args:
            key: 配置键，例如 "network.hfe.enabled"
            default: 默认值
        
        Returns:
            配置值
        
        Example:
            >>> cfg.get("network.hfe.enabled", False)
            True
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置项（支持点分隔的路径）
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key: str) -> Any:
        """字典式访问"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._config
    
    @property
    def config(self) -> Dict:
        """返回原始配置字典"""
        return self._config
    
    # ==================== 便捷属性 ====================
    
    @property
    def experiment_id(self) -> str:
        """实验ID"""
        return self.get('experiment.id', 'default')
    
    @property
    def hfe_enabled(self) -> bool:
        """是否启用HFE模块"""
        return self.get('network.hfe.enabled', False)
    
    @property
    def cdag_enabled(self) -> bool:
        """是否启用CDAG模块"""
        return self.get('network.cdag.enabled', False)
    
    @property
    def hcl_enabled(self) -> bool:
        """是否启用HCL损失"""
        return self.get('loss.hcl.enabled', False)
    
    @property
    def hcl_weight(self) -> float:
        """HCL损失权重"""
        return self.get('loss.hcl.weight', 0.1)
    
    @property
    def n_classes(self) -> int:
        """语义类别数"""
        return self.get('tasks.semantic_segmentation.n_classes', 6)
    
    @property
    def backbone(self) -> str:
        """骨干网络名称"""
        return self.get('network.backbone', 'MinkUNet14A')
    
    @property
    def learning_rate(self) -> float:
        """学习率"""
        return self.get('train.lr', 0.005)
    
    @property
    def max_epochs(self) -> int:
        """最大训练轮数"""
        return self.get('train.max_epoch', 500)
    
    @property
    def batch_size(self) -> int:
        """批大小"""
        return self.get('train.batch_size', 1)
    
    @property
    def voxel_resolution(self) -> float:
        """体素分辨率"""
        return self.get('train.voxel_resolution', 0.003)
    
    def validate(self) -> bool:
        """
        验证配置完整性
        
        Returns:
            配置是否有效
        """
        errors = []
        
        # 必需的配置项检查
        required_keys = [
            'experiment.id',
            'data.path',
            'network.backbone',
            'tasks.semantic_segmentation.n_classes',
            'train.max_epoch',
            'train.lr'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                errors.append(f"缺少必需的配置项: {key}")
        
        # 数值范围检查
        if self.learning_rate <= 0:
            errors.append("学习率必须为正数")
        
        if self.max_epochs <= 0:
            errors.append("最大训练轮数必须为正整数")
        
        if self.n_classes <= 0:
            errors.append("类别数必须为正整数")
        
        # HFE配置检查
        if self.hfe_enabled:
            if not self.get('network.hfe.global_branch'):
                errors.append("HFE启用时必须配置global_branch")
            if not self.get('network.hfe.semantic_branch'):
                errors.append("HFE启用时必须配置semantic_branch")
            if not self.get('network.hfe.local_branch'):
                errors.append("HFE启用时必须配置local_branch")
        
        # CDAG配置检查
        if self.cdag_enabled:
            if not self.get('network.cdag.apply_to'):
                errors.append("CDAG启用时必须配置apply_to")
        
        # HCL配置检查
        if self.hcl_enabled:
            if self.hcl_weight <= 0:
                errors.append("HCL启用时weight必须为正数")
        
        if errors:
            for error in errors:
                print(f"[配置错误] {error}")
            return False
        
        return True
    
    def summary(self) -> str:
        """
        生成配置摘要
        
        Returns:
            配置摘要字符串
        """
        lines = [
            "=" * 60,
            f"实验配置摘要: {self.experiment_id}",
            "=" * 60,
            f"描述: {self.get('experiment.description', 'N/A')}",
            "",
            "【模块状态】",
            f"  - HFE: {'✓ 启用' if self.hfe_enabled else '✗ 禁用'}",
            f"  - CDAG: {'✓ 启用' if self.cdag_enabled else '✗ 禁用'}",
            f"  - HCL: {'✓ 启用 (λ={self.hcl_weight})' if self.hcl_enabled else '✗ 禁用'}",
            "",
            "【网络配置】",
            f"  - 骨干网络: {self.backbone}",
            f"  - 类别数: {self.n_classes}",
            f"  - 跳跃连接: {self.get('network.skip', 'full')}",
            "",
            "【训练配置】",
            f"  - 最大轮数: {self.max_epochs}",
            f"  - 学习率: {self.learning_rate}",
            f"  - 批大小: {self.batch_size}",
            f"  - 体素分辨率: {self.voxel_resolution}",
            "",
            "=" * 60
        ]
        return '\n'.join(lines)
    
    def save(self, save_path: str):
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"配置已保存至: {save_path}")


def load_config(config_path: str, **overrides) -> ConfigManager:
    """
    加载配置的便捷函数
    
    Args:
        config_path: 配置文件路径
        **overrides: 配置覆盖项
    
    Returns:
        ConfigManager实例
    
    Example:
        >>> cfg = load_config("config/config_ours.yaml", train={'lr': 0.001})
        >>> print(cfg.learning_rate)
        0.001
    """
    return ConfigManager(config_path=config_path, config_dict=overrides if overrides else None)


def merge_configs(*config_paths: str) -> ConfigManager:
    """
    合并多个配置文件
    
    后面的配置会覆盖前面的配置
    
    Args:
        *config_paths: 配置文件路径列表
    
    Returns:
        ConfigManager实例
    """
    merged = ConfigManager()
    for path in config_paths:
        with open(path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
        if file_config:
            merged._deep_update(merged._config, file_config)
    return merged


# ==================== 预定义配置获取函数 ====================

def get_ablation_configs() -> Dict[str, str]:
    """
    获取模块级消融实验配置路径
    
    Returns:
        配置名称到路径的映射
    """
    config_dir = Path(__file__).parent.parent / 'config'
    return {
        'baseline': str(config_dir / 'config_baseline.yaml'),
        'hfe_only': str(config_dir / 'config_ablation_hfe.yaml'),
        'cdag_only': str(config_dir / 'config_ablation_cdag.yaml'),
        'hcl_only': str(config_dir / 'config_ablation_hcl.yaml'),
        'hfe_cdag': str(config_dir / 'config_ablation_hfe_cdag.yaml'),
        'ours': str(config_dir / 'config_ours.yaml'),
    }


def get_cdag_internal_ablation_configs() -> Dict[str, str]:
    """
    获取CDAG模块内部消融实验配置 (对应论文 Tab. ablation_cdag)
    
    消融维度: 空间 -> 空间+通道 -> 空间+通道+多尺度 -> 完整四维度
    
    Returns:
        配置名称到路径的映射
    """
    config_dir = Path(__file__).parent.parent / 'config' / 'ablation_cdag'
    return {
        'cdag_spatial': str(config_dir / 'config_cdag_spatial.yaml'),
        'cdag_spatial_channel': str(config_dir / 'config_cdag_spatial_channel.yaml'),
        'cdag_spatial_channel_multiscale': str(config_dir / 'config_cdag_spatial_channel_multiscale.yaml'),
        'cdag_full': str(config_dir / 'config_cdag_full.yaml'),
    }


def get_hfe_internal_ablation_configs() -> Dict[str, str]:
    """
    获取HFE模块内部消融实验配置 (对应论文 Tab. ablation_hfe)
    
    消融设计: 仅膨胀率差异 vs 完整专门化设计
    
    Returns:
        配置名称到路径的映射
    """
    config_dir = Path(__file__).parent.parent / 'config' / 'ablation_hfe'
    return {
        'hfe_dilation_only': str(config_dir / 'config_hfe_dilation_only.yaml'),
        'hfe_full': str(config_dir / 'config_hfe_full.yaml'),
    }


def get_cdag_position_ablation_configs() -> Dict[str, str]:
    """
    获取CDAG应用位置消融实验配置 (对应论文 Tab. ablation_cdag_pos)
    
    消融位置: 语义/树木/实例解码器的不同组合
    
    Returns:
        配置名称到路径的映射
    """
    config_dir = Path(__file__).parent.parent / 'config' / 'ablation_cdag_pos'
    return {
        'cdag_pos_sem': str(config_dir / 'config_cdag_pos_sem.yaml'),
        'cdag_pos_tree': str(config_dir / 'config_cdag_pos_tree.yaml'),
        'cdag_pos_inst': str(config_dir / 'config_cdag_pos_inst.yaml'),
        'cdag_pos_all': str(config_dir / 'config_cdag_pos_all.yaml'),
    }


def get_hcl_weight_ablation_configs() -> Dict[str, str]:
    """
    获取HCL损失权重消融实验配置 (对应论文 Tab. ablation_hcl)
    
    消融权重: λ = 0, 0.05, 0.1, 0.2, 0.5
    
    Returns:
        配置名称到路径的映射
    """
    config_dir = Path(__file__).parent.parent / 'config' / 'ablation_hcl'
    return {
        'hcl_lambda_0': str(config_dir / 'config_hcl_lambda_0.yaml'),
        'hcl_lambda_0.05': str(config_dir / 'config_hcl_lambda_0.05.yaml'),
        'hcl_lambda_0.1': str(config_dir / 'config_hcl_lambda_0.1.yaml'),
        'hcl_lambda_0.2': str(config_dir / 'config_hcl_lambda_0.2.yaml'),
        'hcl_lambda_0.5': str(config_dir / 'config_hcl_lambda_0.5.yaml'),
    }


def get_all_ablation_configs() -> Dict[str, Dict[str, str]]:
    """
    获取所有消融实验配置（按类别分组）
    
    Returns:
        分类的配置字典
    """
    return {
        'module_level': get_ablation_configs(),
        'cdag_internal': get_cdag_internal_ablation_configs(),
        'hfe_internal': get_hfe_internal_ablation_configs(),
        'cdag_position': get_cdag_position_ablation_configs(),
        'hcl_weight': get_hcl_weight_ablation_configs(),
    }


def print_config_comparison():
    """打印所有配置的对比表格"""
    
    # 模块级消融
    print("\n" + "=" * 70)
    print("【1】模块级消融实验 (对应论文 Tab. ablation_modules)")
    print("=" * 70)
    configs = get_ablation_configs()
    print(f"{'配置名称':<15} {'HFE':<8} {'CDAG':<8} {'HCL':<8} {'描述':<25}")
    print("-" * 70)
    
    for name, path in configs.items():
        try:
            cfg = load_config(path)
            hfe = '✓' if cfg.hfe_enabled else '✗'
            cdag = '✓' if cfg.cdag_enabled else '✗'
            hcl = '✓' if cfg.hcl_enabled else '✗'
            desc = cfg.get('experiment.description', '')[:25]
            print(f"{name:<15} {hfe:<8} {cdag:<8} {hcl:<8} {desc:<25}")
        except Exception as e:
            print(f"{name:<15} 加载失败: {e}")
    
    # CDAG内部消融
    print("\n" + "=" * 70)
    print("【2】CDAG设计选择消融 (对应论文 Tab. ablation_cdag)")
    print("=" * 70)
    configs = get_cdag_internal_ablation_configs()
    print(f"{'配置名称':<35} {'空间':<6} {'通道':<6} {'多尺度':<8} {'像素':<6}")
    print("-" * 70)
    
    for name, path in configs.items():
        try:
            cfg = load_config(path)
            spatial = '✓' if cfg.get('network.cdag.spatial_gate.enabled', False) else '✗'
            channel = '✓' if cfg.get('network.cdag.channel_attention.enabled', False) else '✗'
            multiscale = '✓' if cfg.get('network.cdag.multiscale_attention.enabled', False) else '✗'
            pixel = '✓' if cfg.get('network.cdag.pixel_attention.enabled', False) else '✗'
            print(f"{name:<35} {spatial:<6} {channel:<6} {multiscale:<8} {pixel:<6}")
        except Exception as e:
            print(f"{name:<35} 加载失败")
    
    # HFE内部消融
    print("\n" + "=" * 70)
    print("【3】HFE分支设计消融 (对应论文 Tab. ablation_hfe)")
    print("=" * 70)
    configs = get_hfe_internal_ablation_configs()
    print(f"{'配置名称':<25} {'全局池化':<10} {'通道重标定':<12} {'多尺度':<8} {'边界增强':<10}")
    print("-" * 70)
    
    for name, path in configs.items():
        try:
            cfg = load_config(path)
            gpool = '✓' if cfg.get('network.hfe.global_branch.use_global_pool', False) else '✗'
            chrecal = '✓' if cfg.get('network.hfe.global_branch.channel_reduction') else '✗'
            multiscale = '✓' if cfg.get('network.hfe.semantic_branch.use_multiscale', True) else '✗'
            edge = '✓' if cfg.get('network.hfe.local_branch.use_edge_enhance', False) else '✗'
            print(f"{name:<25} {gpool:<10} {chrecal:<12} {multiscale:<8} {edge:<10}")
        except Exception as e:
            print(f"{name:<25} 加载失败")
    
    # CDAG位置消融
    print("\n" + "=" * 70)
    print("【4】CDAG应用位置消融 (对应论文 Tab. ablation_cdag_pos)")
    print("=" * 70)
    configs = get_cdag_position_ablation_configs()
    print(f"{'配置名称':<20} {'语义解码器':<12} {'树木解码器':<12} {'实例解码器':<12}")
    print("-" * 70)
    
    for name, path in configs.items():
        try:
            cfg = load_config(path)
            sem = '✓' if cfg.get('network.cdag.apply_to.semantic_decoder', False) else '✗'
            tree = '✓' if cfg.get('network.cdag.apply_to.tree_decoder', False) else '✗'
            inst = '✓' if cfg.get('network.cdag.apply_to.instance_decoder', False) else '✗'
            print(f"{name:<20} {sem:<12} {tree:<12} {inst:<12}")
        except Exception as e:
            print(f"{name:<20} 加载失败")
    
    # HCL权重消融
    print("\n" + "=" * 70)
    print("【5】HCL损失权重消融 (对应论文 Tab. ablation_hcl)")
    print("=" * 70)
    configs = get_hcl_weight_ablation_configs()
    print(f"{'配置名称':<25} {'λ值':<10} {'HCL启用':<10}")
    print("-" * 70)
    
    for name, path in configs.items():
        try:
            cfg = load_config(path)
            weight = cfg.get('loss.hcl.weight', 0)
            enabled = '✓' if cfg.get('loss.hcl.enabled', False) else '✗'
            print(f"{name:<25} {weight:<10} {enabled:<10}")
        except Exception as e:
            print(f"{name:<25} 加载失败")
    
    print("=" * 70 + "\n")


if __name__ == '__main__':
    # 测试配置管理器
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        cfg = load_config(config_path)
        print(cfg.summary())
        cfg.validate()
    else:
        print("使用方法: python config_manager.py <config_path>")
        print("\n显示配置对比表:")
        print_config_comparison()
