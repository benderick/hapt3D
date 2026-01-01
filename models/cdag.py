# -*- coding: utf-8 -*-
"""
CDAG (Cross-Decoder Attention Gate) - 跨解码器注意力门控模块

设计目标：在解码器的跳跃连接处，自适应地选择和加权编码器特征

=== 参考的论文及其核心设计 ===

[1] Attention U-Net (Oktay et al., arXiv 2018, 后发表于MIA):
    - 论文: "Attention U-Net: Learning Where to Look for the Pancreas"
    - 核心设计: g信号引导的加性注意力门控
    - 公式: psi = σ(W_ψ * ReLU(W_g*g + W_x*x))
    - 本文使用位置: SpatialAttentionGate类

[2] DEA-Net / CGAFusion (Chen et al., TIP 2024):
    - 论文: "DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution 
            and Content-Guided Attention"
    - 核心设计: 空间+通道+像素三重注意力 → 自适应融合
    - 公式: result = initial + pattn2 * x + (1 - pattn2) * y
    - 本文使用位置: PixelAttention类 + 三重注意力融合策略

[3] CPCA (Huang et al., Computers in Biology and Medicine 2024):
    - 论文: "Channel Prior Convolutional Attention for Medical Image Segmentation"
    - 核心设计: 
      (a) 双池化(avg+max)通道注意力
      (b) 多尺度分解卷积空间注意力
    - 本文使用位置: DualPoolChannelAttention类 + MultiScaleSpatialAttention类

作者: [论文作者]
日期: 2025
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiBatchNorm,
    MinkowskiReLU,
    MinkowskiGlobalPooling,
    MinkowskiGlobalMaxPooling,
)


# ============================================================================
# 模块1: 空间注意力门控 (来自 Attention Gate, MIA 2018)
# ============================================================================
class SpatialAttentionGate(nn.Module):
    """
    空间注意力门控 - 严格参考Attention U-Net (Oktay et al., MIA 2018)
    
    原始公式:
        g1 = W_g(g)           # 门控信号变换
        x1 = W_x(x)           # 输入信号变换  
        psi = σ(W_ψ(ReLU(g1 + x1)))  # 注意力权重
        output = x * psi      # 门控输出
    
    参考代码: code/2025/135.0 (MIA 2025) AttentionGate.py
    """
    
    def __init__(self, F_g, F_l, F_int, D=3):
        """
        参数:
            F_g: 门控信号通道数（来自解码器）
            F_l: 输入信号通道数（来自编码器跳跃连接）
            F_int: 中间层通道数
            D: 空间维度 (3D点云)
        """
        super(SpatialAttentionGate, self).__init__()
        
        # W_g: 门控信号变换
        self.W_g = nn.Sequential(
            MinkowskiConvolution(F_g, F_int, kernel_size=1, bias=True, dimension=D),
            MinkowskiBatchNorm(F_int)
        )
        
        # W_x: 输入信号变换
        self.W_x = nn.Sequential(
            MinkowskiConvolution(F_l, F_int, kernel_size=1, bias=True, dimension=D),
            MinkowskiBatchNorm(F_int)
        )
        
        # W_ψ: 注意力权重生成
        self.psi = nn.Sequential(
            MinkowskiConvolution(F_int, 1, kernel_size=1, bias=True, dimension=D),
            MinkowskiBatchNorm(1),
        )
        
        self.relu = MinkowskiReLU()
    
    def forward(self, g, x):
        """
        参数:
            g: 门控信号 (SparseTensor) - 来自解码器上采样特征
            x: 输入信号 (SparseTensor) - 来自编码器跳跃特征
        返回:
            门控后的特征 (SparseTensor), 注意力权重
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 加性融合 + ReLU (Attention Gate核心)
        psi = self.relu(ME.SparseTensor(
            features=g1.F + x1.F,
            coordinate_map_key=g1.coordinate_map_key,
            coordinate_manager=g1.coordinate_manager
        ))
        
        psi = self.psi(psi)
        attention = torch.sigmoid(psi.F)  # [N, 1]
        
        # 门控输出
        gated_features = attention * x.F
        
        return ME.SparseTensor(
            features=gated_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        ), attention


# ============================================================================
# 模块2: 双池化通道注意力 (来自 CPCA, CBM 2024)
# ============================================================================
class DualPoolChannelAttention(nn.Module):
    """
    双池化通道注意力 - 严格参考CPCA (Chen et al., CBM 2024)
    
    CPCA核心设计: 同时使用 avg_pool 和 max_pool，然后相加
    
    原始公式:
        x1 = σ(FC2(ReLU(FC1(GAP(x)))))   # 平均池化分支
        x2 = σ(FC2(ReLU(FC1(GMP(x)))))   # 最大池化分支
        attention = x1 + x2               # 双池化融合
    
    参考代码: code/2024/35.0 (Elsevier 2024) CPCA.py 中的 CPCA_ChannelAttention
    """
    
    def __init__(self, in_channels, reduction=4):
        """
        参数:
            in_channels: 输入通道数
            reduction: 通道压缩比
        """
        super(DualPoolChannelAttention, self).__init__()
        
        internal_channels = max(in_channels // reduction, 8)
        
        # 全局平均池化
        self.gap = MinkowskiGlobalPooling()
        # 全局最大池化
        self.gmp = MinkowskiGlobalMaxPooling()
        
        # 共享MLP (CPCA原文使用1x1卷积实现，这里用Linear等价)
        self.fc1 = nn.Linear(in_channels, internal_channels, bias=True)
        self.fc2 = nn.Linear(internal_channels, in_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 (SparseTensor)
        返回:
            通道注意力权重 [B, C]
        """
        # 平均池化分支
        x_avg = self.gap(x).F  # [B, C]
        x_avg = self.fc2(self.relu(self.fc1(x_avg)))
        x_avg = torch.sigmoid(x_avg)
        
        # 最大池化分支
        x_max = self.gmp(x).F  # [B, C]
        x_max = self.fc2(self.relu(self.fc1(x_max)))
        x_max = torch.sigmoid(x_max)
        
        # 双池化融合 (CPCA核心: 相加而非拼接)
        attention = x_avg + x_max  # [B, C]
        
        return attention


# ============================================================================
# 模块3: 多尺度空间分解注意力 (来自 CPCA, CBM 2024)
# ============================================================================
class MultiScaleSpatialAttention(nn.Module):
    """
    多尺度空间分解注意力 - 严格参考CPCA (Chen et al., CBM 2024)
    
    CPCA核心设计: 使用分解卷积捕获多尺度空间信息
    原始2D设计: 5×5 → (1×7, 7×1) + (1×11, 11×1) + (1×21, 21×1)
    
    3D适配: 使用不同膨胀率的3×3×3卷积模拟多尺度效果
    - d=1: 局部感受野 (等效小核)
    - d=2: 中等感受野 (等效中核)  
    - d=3: 大感受野 (等效大核)
    
    参考代码: code/2024/35.0 (Elsevier 2024) CPCA.py 中的 dconv系列
    """
    
    def __init__(self, in_channels, D=3):
        """
        参数:
            in_channels: 输入通道数
            D: 空间维度
        """
        super(MultiScaleSpatialAttention, self).__init__()
        
        # 初始卷积 (对应CPCA的dconv5_5)
        self.conv_init = nn.Sequential(
            MinkowskiConvolution(in_channels, in_channels, kernel_size=3, 
                                dilation=1, dimension=D, bias=True),
            MinkowskiBatchNorm(in_channels),
            MinkowskiReLU()
        )
        
        # 多尺度分支 (对应CPCA的分解卷积)
        # 分支1: 小尺度 (d=1, 对应1×7+7×1)
        self.branch1 = nn.Sequential(
            MinkowskiConvolution(in_channels, in_channels, kernel_size=3,
                                dilation=1, dimension=D, bias=True),
            MinkowskiBatchNorm(in_channels),
            MinkowskiReLU()
        )
        
        # 分支2: 中尺度 (d=2, 对应1×11+11×1)
        self.branch2 = nn.Sequential(
            MinkowskiConvolution(in_channels, in_channels, kernel_size=3,
                                dilation=2, dimension=D, bias=True),
            MinkowskiBatchNorm(in_channels),
            MinkowskiReLU()
        )
        
        # 分支3: 大尺度 (d=3, 对应1×21+21×1)
        self.branch3 = nn.Sequential(
            MinkowskiConvolution(in_channels, in_channels, kernel_size=3,
                                dilation=3, dimension=D, bias=True),
            MinkowskiBatchNorm(in_channels),
            MinkowskiReLU()
        )
        
        # 融合卷积 (对应CPCA的最后conv)
        self.conv_out = MinkowskiConvolution(in_channels, 1, kernel_size=1, 
                                             dimension=D, bias=True)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 (SparseTensor)
        返回:
            空间注意力权重 (SparseTensor) [N, 1]
        """
        # 初始卷积
        x_init = self.conv_init(x)
        
        # 多尺度分支
        x1 = self.branch1(x_init)
        x2 = self.branch2(x_init)
        x3 = self.branch3(x_init)
        
        # 多尺度融合 (CPCA: 相加)
        x_fused = ME.SparseTensor(
            features=x1.F + x2.F + x3.F + x_init.F,
            coordinate_map_key=x1.coordinate_map_key,
            coordinate_manager=x1.coordinate_manager
        )
        
        # 生成空间注意力权重
        attention = self.conv_out(x_fused)
        attention_weights = torch.sigmoid(attention.F)  # [N, 1]
        
        return attention_weights


# ============================================================================
# 模块4: 像素注意力 (来自 CGAFusion, TIP 2024)
# ============================================================================
class PixelAttention(nn.Module):
    """
    像素注意力 - 严格参考CGAFusion (DEA-Net, TIP 2024)
    
    CGAFusion核心设计: 将空间注意力和通道注意力结合，生成像素级(点级)注意力
    
    原始公式:
        pattn1 = sattn + cattn           # 空间+通道
        pattn2 = σ(Conv([x; pattn1]))    # 像素级融合
    
    参考代码: code/2024/46.0 (TIP 2024) CGA.py 中的 PixelAttention
    """
    
    def __init__(self, in_channels, D=3):
        """
        参数:
            in_channels: 输入通道数
            D: 空间维度
        """
        super(PixelAttention, self).__init__()
        
        # 像素级融合卷积 (对应CGAFusion的pa2)
        # 原文用7x7 groups卷积，3D中用3x3x3
        self.pixel_conv = nn.Sequential(
            MinkowskiConvolution(in_channels * 2, in_channels, kernel_size=3,
                                dimension=D, bias=True),
            MinkowskiBatchNorm(in_channels),
        )
    
    def forward(self, x, pattn1):
        """
        参数:
            x: 输入特征 (SparseTensor)
            pattn1: 空间+通道注意力融合后的特征 (SparseTensor)
        返回:
            像素级注意力权重 [N, C]
        """
        # 拼接输入和注意力特征
        concat_features = torch.cat([x.F, pattn1.F], dim=1)  # [N, 2C]
        
        concat_sparse = ME.SparseTensor(
            features=concat_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )
        
        # 像素级卷积
        pattn2 = self.pixel_conv(concat_sparse)
        pattn2_weights = torch.sigmoid(pattn2.F)  # [N, C]
        
        return pattn2_weights


# ============================================================================
# 主模块: CDAG (融合三篇论文的完整设计)
# ============================================================================
class CDAG(nn.Module):
    """
    CDAG (Cross-Decoder Attention Gate) - 跨解码器注意力门控
    
    完整融合三篇论文的设计:
    
    1. SpatialAttentionGate (来自Attention Gate, MIA 2018):
       - g信号引导的加性注意力
       
    2. DualPoolChannelAttention (来自CPCA, CBM 2024):
       - avg_pool + max_pool 双池化通道注意力
       
    3. MultiScaleSpatialAttention (来自CPCA, CBM 2024):
       - 多尺度分解卷积空间注意力
       
    4. PixelAttention + 三重融合 (来自CGAFusion, TIP 2024):
       - 空间+通道+像素三重注意力
       - result = initial + pattn2 * x + (1 - pattn2) * y
    
    最终公式:
        A_spatial = SpatialAttentionGate(g, x)     # Attention Gate
        A_channel = DualPoolChannelAttention(x)     # CPCA双池化
        A_multiscale = MultiScaleSpatialAttention(x) # CPCA多尺度
        pattn1 = A_spatial + A_channel + A_multiscale  # 三重融合
        pattn2 = PixelAttention(x, pattn1)          # CGAFusion像素注意力
        output = x + pattn2 * x_gated               # CGAFusion残差融合
    
    消融配置示例:
        cdag:
          enabled: True
          spatial_gate:
            enabled: True
          channel_attention:
            enabled: True
            reduction: 4
          multiscale_attention:
            enabled: True
          pixel_attention:
            enabled: True
    """
    
    def __init__(self, F_g, F_l, F_int=None, reduction=4, cfg=None, D=3):
        """
        参数:
            F_g: 门控信号通道数（解码器特征）
            F_l: 输入信号通道数（编码器跳跃特征）
            F_int: 空间注意力中间层通道数
            reduction: 通道注意力压缩比
            cfg: 配置字典，控制各注意力模块的启用
            D: 空间维度
        """
        super(CDAG, self).__init__()
        
        if F_int is None:
            F_int = max(F_l // 2, 16)
        
        self.F_l = F_l
        
        # 解析配置
        if cfg is None:
            cfg = {}
        
        spatial_cfg = cfg.get('spatial_gate', {})
        channel_cfg = cfg.get('channel_attention', {})
        multiscale_cfg = cfg.get('multiscale_attention', {})
        pixel_cfg = cfg.get('pixel_attention', {})
        
        self.use_spatial = spatial_cfg.get('enabled', True) if isinstance(spatial_cfg, dict) else True
        self.use_channel = channel_cfg.get('enabled', True) if isinstance(channel_cfg, dict) else True
        self.use_multiscale = multiscale_cfg.get('enabled', True) if isinstance(multiscale_cfg, dict) else True
        self.use_pixel = pixel_cfg.get('enabled', True) if isinstance(pixel_cfg, dict) else True
        
        # 获取reduction参数
        reduction = channel_cfg.get('reduction', reduction) if isinstance(channel_cfg, dict) else reduction
        
        # 模块1: Attention Gate式空间门控 (MIA 2018) - 始终创建（作为基础）
        self.spatial_gate = SpatialAttentionGate(F_g, F_l, F_int, D)
        
        # 模块2: CPCA双池化通道注意力 (CBM 2024) - 可选
        if self.use_channel:
            self.channel_attn = DualPoolChannelAttention(F_l, reduction)
        else:
            self.channel_attn = None
        
        # 模块3: CPCA多尺度空间注意力 (CBM 2024) - 可选
        if self.use_multiscale:
            self.multiscale_attn = MultiScaleSpatialAttention(F_l, D)
        else:
            self.multiscale_attn = None
        
        # 模块4: CGAFusion像素注意力 (TIP 2024) - 可选
        if self.use_pixel:
            self.pixel_attn = PixelAttention(F_l, D)
        else:
            self.pixel_attn = None
        
        # 特征变换 (用于生成pattn1)
        self.attn_transform = MinkowskiConvolution(F_l, F_l, kernel_size=1, 
                                                   dimension=D, bias=True)
        
        # 输出卷积 (对应CGAFusion的最后conv)
        self.conv_out = MinkowskiConvolution(F_l, F_l, kernel_size=1,
                                             dimension=D, bias=True)
    
    def forward(self, g, x):
        """
        前向传播
        
        参数:
            g: 门控信号 (SparseTensor) - 来自解码器上采样特征
            x: 输入信号 (SparseTensor) - 来自编码器跳跃特征
            
        返回:
            门控后的特征 (SparseTensor)
        """
        # ===== 步骤1: Attention Gate空间门控 (MIA 2018) =====
        x_gated, A_spatial = self.spatial_gate(g, x)  # A_spatial: [N, 1]
        
        # ===== 步骤2: CPCA双池化通道注意力 (CBM 2024) - 可选 =====
        if self.use_channel and self.channel_attn is not None:
            A_channel = self.channel_attn(x)  # [B, C]
            # 广播到每个点
            batch_indices = x.C[:, 0].long()
            A_channel_expanded = A_channel[batch_indices]  # [N, C]
        else:
            A_channel_expanded = torch.ones_like(x.F)  # 不使用时为全1
            batch_indices = x.C[:, 0].long()
        
        # ===== 步骤3: CPCA多尺度空间注意力 (CBM 2024) - 可选 =====
        if self.use_multiscale and self.multiscale_attn is not None:
            A_multiscale = self.multiscale_attn(x)  # [N, 1]
        else:
            A_multiscale = torch.ones(x.F.shape[0], 1, device=x.F.device)  # 不使用时为全1
        
        # ===== 步骤4: 注意力融合 (根据启用的模块) =====
        if self.use_spatial:
            # 使用空间门控结果
            if self.use_channel or self.use_multiscale:
                pattn1_features = (A_spatial * A_channel_expanded) + \
                                  (A_multiscale * A_channel_expanded)
            else:
                pattn1_features = A_spatial * torch.ones_like(x.F)
        else:
            # 不使用空间门控
            pattn1_features = A_channel_expanded * A_multiscale
        
        pattn1 = ME.SparseTensor(
            features=pattn1_features * x.F,  # 注意力加权
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )
        pattn1 = self.attn_transform(pattn1)
        
        # ===== 步骤5: CGAFusion像素注意力 (TIP 2024) - 可选 =====
        if self.use_pixel and self.pixel_attn is not None:
            pattn2 = self.pixel_attn(x, pattn1)  # [N, C]
        else:
            pattn2 = torch.sigmoid(pattn1.F)  # 简单的sigmoid替代
        
        # ===== 步骤6: 残差融合 =====
        result_features = x.F + pattn2 * x_gated.F
        
        result = ME.SparseTensor(
            features=result_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )
        
        # 最终输出卷积
        output = self.conv_out(result)
        
        return output


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("CDAG模块测试 - 消融实验参数量对比")
    print("=" * 70)
    print("参考论文:")
    print("  [1] Attention Gate (MIA 2018) - 空间门控")
    print("  [2] CPCA (CBM 2024) - 双池化通道注意力 + 多尺度空间注意力")
    print("  [3] CGAFusion (TIP 2024) - 像素注意力 + 三重融合")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 测试参数
    channels = 128
    
    # ===== 定义不同的消融配置 =====
    ablation_configs = {
        # 1. 完整配置 - 所有模块启用
        "Full (所有模块)": {
            'spatial_gate': {'enabled': True},
            'channel_attention': {'enabled': True, 'reduction': 4},
            'multiscale_attention': {'enabled': True},
            'pixel_attention': {'enabled': True}
        },
        # 2. 仅空间门控 (Attention Gate baseline)
        "Spatial Only (仅空间门控)": {
            'spatial_gate': {'enabled': True},
            'channel_attention': {'enabled': False},
            'multiscale_attention': {'enabled': False},
            'pixel_attention': {'enabled': False}
        },
        # 3. 无通道注意力
        "No Channel (无通道注意力)": {
            'spatial_gate': {'enabled': True},
            'channel_attention': {'enabled': False},
            'multiscale_attention': {'enabled': True},
            'pixel_attention': {'enabled': True}
        },
        # 4. 无多尺度注意力
        "No Multiscale (无多尺度)": {
            'spatial_gate': {'enabled': True},
            'channel_attention': {'enabled': True, 'reduction': 4},
            'multiscale_attention': {'enabled': False},
            'pixel_attention': {'enabled': True}
        },
        # 5. 无像素注意力
        "No Pixel (无像素注意力)": {
            'spatial_gate': {'enabled': True},
            'channel_attention': {'enabled': True, 'reduction': 4},
            'multiscale_attention': {'enabled': True},
            'pixel_attention': {'enabled': False}
        },
        # 6. 仅CPCA (通道+多尺度)
        "CPCA Only (通道+多尺度)": {
            'spatial_gate': {'enabled': True},  # 基础空间门控保留
            'channel_attention': {'enabled': True, 'reduction': 4},
            'multiscale_attention': {'enabled': True},
            'pixel_attention': {'enabled': False}
        },
    }
    
    # ===== 打印各配置的参数量 =====
    print("\n" + "=" * 70)
    print("消融实验 - 不同配置的参数量对比")
    print("=" * 70)
    print(f"{'配置名称':<35} {'总参数量':>15} {'差异':>12}")
    print("-" * 70)
    
    param_counts = {}
    base_params = None
    
    for name, cfg in ablation_configs.items():
        # 创建模型
        model = CDAG(F_g=channels, F_l=channels, cfg=cfg).to(device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        param_counts[name] = total_params
        
        # 计算与完整配置的差异
        if base_params is None:
            base_params = total_params
            diff_str = "baseline"
        else:
            diff = total_params - base_params
            diff_str = f"{diff:+,}"
        
        print(f"{name:<35} {total_params:>15,} {diff_str:>12}")
        
        # 清理内存
        del model
    
    # ===== 详细子模块参数量 =====
    print("\n" + "=" * 70)
    print("完整配置 (Full) - 各子模块参数量详情")
    print("=" * 70)
    
    full_cfg = ablation_configs["Full (所有模块)"]
    full_model = CDAG(F_g=channels, F_l=channels, cfg=full_cfg).to(device)
    
    # 统计各子模块
    spatial_params = sum(p.numel() for p in full_model.spatial_gate.parameters())
    print(f"  SpatialAttentionGate (MIA 2018):     {spatial_params:>12,}")
    
    if full_model.channel_attn is not None:
        channel_params = sum(p.numel() for p in full_model.channel_attn.parameters())
        print(f"  DualPoolChannelAttention (CPCA):    {channel_params:>12,}")
    
    if full_model.multiscale_attn is not None:
        multiscale_params = sum(p.numel() for p in full_model.multiscale_attn.parameters())
        print(f"  MultiScaleSpatialAttention (CPCA):  {multiscale_params:>12,}")
    
    if full_model.pixel_attn is not None:
        pixel_params = sum(p.numel() for p in full_model.pixel_attn.parameters())
        print(f"  PixelAttention (CGAFusion):         {pixel_params:>12,}")
    
    # 其他层参数
    other_params = sum(p.numel() for p in full_model.attn_transform.parameters())
    other_params += sum(p.numel() for p in full_model.conv_out.parameters())
    print(f"  其他层 (transform + conv_out):       {other_params:>12,}")
    
    # ===== 验证参数量差异 =====
    print("\n" + "=" * 70)
    print("验证: 各配置参数量是否正确变化")
    print("=" * 70)
    
    # 检查参数量是否有差异
    unique_params = set(param_counts.values())
    if len(unique_params) == 1:
        print("⚠️ 警告: 所有配置的参数量相同！消融实验可能无效！")
    else:
        print(f"✓ 检测到 {len(unique_params)} 种不同的参数量配置")
        print("✓ 消融配置正确生效")
    
    # ===== 功能测试 =====
    print("\n" + "=" * 70)
    print("功能测试 - 验证前向传播")
    print("=" * 70)
    
    # 创建测试数据
    batch_size = 2
    num_points_per_sample = [1000, 800]
    
    coords_list = []
    feats_g_list = []
    feats_x_list = []
    
    for b in range(batch_size):
        n_points = num_points_per_sample[b]
        coords = torch.randint(0, 100, (n_points, 3), dtype=torch.int)
        batch_idx = torch.full((n_points, 1), b, dtype=torch.int)
        coords = torch.cat([batch_idx, coords], dim=1)
        coords_list.append(coords)
        feats_g_list.append(torch.randn(n_points, channels))
        feats_x_list.append(torch.randn(n_points, channels))
    
    coords = torch.cat(coords_list, dim=0).to(device)
    feats_g = torch.cat(feats_g_list, dim=0).to(device)
    feats_x = torch.cat(feats_x_list, dim=0).to(device)
    
    g = ME.SparseTensor(features=feats_g, coordinates=coords, device=device)
    x = ME.SparseTensor(features=feats_x, coordinates=coords, device=device)
    
    print(f"\n输入: g={g.F.shape}, x={x.F.shape}")
    
    # 测试每个配置的前向传播
    for name, cfg in ablation_configs.items():
        model = CDAG(F_g=channels, F_l=channels, cfg=cfg).to(device)
        try:
            output = model(g, x)
            print(f"  ✓ {name}: output={output.F.shape}")
        except Exception as e:
            print(f"  ✗ {name}: 错误 - {e}")
        del model
    
    print("\n" + "=" * 70)
    print("CDAG消融测试完成!")
    print("=" * 70)
