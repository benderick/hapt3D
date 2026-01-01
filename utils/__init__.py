# -*- coding: utf-8 -*-
"""
Utils模块 - 工具函数和损失函数
"""

from .hcl_loss import HierarchicalConsistencyLoss, HCLv2, hcl_loss
from .config_manager import (
    ConfigManager, 
    load_config, 
    merge_configs, 
    get_ablation_configs,
    get_cdag_internal_ablation_configs,
    get_hfe_internal_ablation_configs,
    get_cdag_position_ablation_configs,
    get_hcl_weight_ablation_configs,
    get_all_ablation_configs,
    print_config_comparison,
)

__all__ = [
    # 损失函数
    'HierarchicalConsistencyLoss',
    'HCLv2', 
    'hcl_loss',
    # 配置管理
    'ConfigManager',
    'load_config',
    'merge_configs',
    'get_ablation_configs',
    'get_cdag_internal_ablation_configs',
    'get_hfe_internal_ablation_configs',
    'get_cdag_position_ablation_configs',
    'get_hcl_weight_ablation_configs',
    'get_all_ablation_configs',
    'print_config_comparison',
]
