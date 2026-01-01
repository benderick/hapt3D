# -*- coding: utf-8 -*-
"""
Models模块 - 网络模型组件

本文提出的三大创新模块:
1. HFE (层次特征增强模块): 从编码器输出生成三种专门化特征
2. CDAG (通道双重注意力门控): 自适应选择跳跃连接特征
3. MinkUNetWithModules: 集成HFE/CDAG的完整网络

使用方法:
    from models import HFE, CDAG, MinkUNet14A_Ours
    from models.hapt3d_ours import HAPT3D
"""

from .hfe import HFE, GlobalContextBranch, SemanticBranch, LocalDetailBranch
from .cdag import CDAG, SpatialAttentionGate, DualPoolChannelAttention, MultiScaleSpatialAttention, PixelAttention
from .minkunet_ours import (
    MinkUNetWithModules,
    MinkUNet14A_Ours,
    MinkUNet18A_Ours,
    MinkUNet34A_Ours,
    MinkUNet14B_Ours,
    MinkUNet18B_Ours,
    MinkUNet34B_Ours,
)

__all__ = [
    # HFE模块
    'HFE',
    'GlobalContextBranch',
    'SemanticBranch', 
    'LocalDetailBranch',
    # CDAG模块
    'CDAG',
    'SpatialAttentionGate',
    'DualPoolChannelAttention',
    'MultiScaleSpatialAttention',
    'PixelAttention',
    # 集成网络
    'MinkUNetWithModules',
    'MinkUNet14A_Ours',
    'MinkUNet18A_Ours',
    'MinkUNet34A_Ours',
    'MinkUNet14B_Ours',
    'MinkUNet18B_Ours',
    'MinkUNet34B_Ours',
]
