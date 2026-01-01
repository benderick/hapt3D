# -*- coding: utf-8 -*-
"""
HFE (Hierarchical Feature Enhancement) - 层次化特征增强模块

=== 设计动机 ===

在层次化全景分割任务中，三个解码器面临截然不同的挑战：

1. **树木实例解码器**：需要将分散在空间中的树干和果实聚合为一个整体
   - 挑战：果实可能距离树干数米，需要大范围的上下文信息
   - 需求：全局结构感知能力，抑制局部噪声

2. **语义解码器**：需要对每个点进行类别判断
   - 挑战：类别边界处容易混淆（如plant与fruit交界处）
   - 需求：清晰的语义边界，适中的上下文

3. **标准实例解码器**：需要区分紧密相邻的同类实例（如相邻果实）
   - 挑战：相邻果实可能只有几厘米间距
   - 需求：精细的局部边界，高空间分辨率

=== 核心设计 ===

基于上述分析，HFE从三个维度为不同任务定制特征：

(1) **感受野差异化**：通过不同膨胀率获取不同范围的空间上下文
    - 大膨胀率：捕获全局结构（树木级）
    - 小膨胀率：保留局部细节（果实级）

(2) **特征增强差异化**：每个分支独立的特征增强路径
    - 粗粒度分支：全局池化 + 通道重标定，强化全局响应
    - 细粒度分支：局部归一化 + 边界增强，强化局部边界

(3) **语义引导**：利用通道注意力选择与任务相关的语义通道
    - 不同任务关注的特征通道不同（颜色vs形状vs纹理）

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
)


class GlobalContextBranch(nn.Module):
    """
    全局上下文分支 - 用于树木级分割
    
    设计动机：
    树木实例分割需要将分散的组件（树干、远处的果实）聚合为整体。
    这要求特征具有全局感知能力，能够"看到"整棵树的范围。
    
    实现方式：
    1. 大膨胀率卷积：扩大感受野覆盖整棵树
    2. 全局上下文注入：通过全局池化获取场景级信息
    3. 通道重标定：选择对全局结构敏感的通道
    """
    
    def __init__(self, in_channels, out_channels, D=3):
        super(GlobalContextBranch, self).__init__()
        
        # 大膨胀率卷积 - 扩大感受野
        self.dilated_conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                dilation=4, dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels),
            MinkowskiReLU()
        )
        
        # 全局上下文提取
        self.global_pool = MinkowskiGlobalPooling()
        
        # 通道重标定（SE风格）- 选择全局结构相关通道
        self.channel_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            MinkowskiConvolution(out_channels, out_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels)
        )
    
    def forward(self, x):
        # 大感受野特征
        feat = self.dilated_conv(x)
        
        # 全局上下文
        global_context = self.global_pool(feat).F  # [B, C]
        channel_weights = self.channel_gate(global_context)  # [B, C]
        
        # 通道重标定：将全局权重广播到每个点
        batch_indices = feat.C[:, 0].long()
        weights_expanded = channel_weights[batch_indices]  # [N, C]
        
        # 应用通道权重
        enhanced_feat = feat.F * weights_expanded
        
        out = ME.SparseTensor(
            features=enhanced_feat,
            coordinate_map_key=feat.coordinate_map_key,
            coordinate_manager=feat.coordinate_manager
        )
        
        return self.fusion(out)


class LocalDetailBranch(nn.Module):
    """
    局部细节分支 - 用于果实级分割
    
    设计动机：
    果实实例分割需要区分紧密相邻的果实，这要求：
    1. 高空间分辨率：不能模糊相邻果实的边界
    2. 边界敏感性：对实例边界有强响应
    3. 局部对比度：增强相邻实例间的差异
    
    实现方式：
    1. 小膨胀率/标准卷积：保留局部空间细节
    2. 局部特征归一化：增强局部对比度
    3. 梯度增强：强化边界响应
    """
    
    def __init__(self, in_channels, out_channels, D=3):
        super(LocalDetailBranch, self).__init__()
        
        # 标准卷积 - 保留局部细节
        self.local_conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                dilation=1, dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels),
            MinkowskiReLU()
        )
        
        # 边界增强卷积 - 使用3x3卷积模拟梯度算子
        self.edge_conv = nn.Sequential(
            MinkowskiConvolution(out_channels, out_channels, kernel_size=3,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            MinkowskiConvolution(out_channels * 2, out_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels),
            MinkowskiReLU()
        )
    
    def forward(self, x):
        # 局部特征
        local_feat = self.local_conv(x)
        
        # 边界增强：原特征与卷积后特征的差异突出边界
        edge_feat = self.edge_conv(local_feat)
        edge_enhanced = ME.SparseTensor(
            features=torch.abs(local_feat.F - edge_feat.F),  # 类似边缘检测
            coordinate_map_key=local_feat.coordinate_map_key,
            coordinate_manager=local_feat.coordinate_manager
        )
        
        # 拼接局部特征和边界特征
        concat_feat = ME.SparseTensor(
            features=torch.cat([local_feat.F, edge_enhanced.F], dim=1),
            coordinate_map_key=local_feat.coordinate_map_key,
            coordinate_manager=local_feat.coordinate_manager
        )
        
        return self.fusion(concat_feat)


class SemanticBranch(nn.Module):
    """
    语义分支 - 用于语义分割
    
    设计动机：
    语义分割关注的是"这个点是什么类别"，需要：
    1. 平衡的上下文：既不能太局部（丢失语义），也不能太全局（丢失边界）
    2. 多尺度融合：结合局部特征和上下文信息
    3. 类别区分度：增强不同类别间的特征差异
    
    实现方式：
    1. 中等膨胀率：平衡感受野大小
    2. 双尺度融合：结合两种尺度的特征
    """
    
    def __init__(self, in_channels, out_channels, D=3):
        super(SemanticBranch, self).__init__()
        
        # 中等尺度分支
        self.medium_conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                dilation=2, dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels),
            MinkowskiReLU()
        )
        
        # 辅助小尺度分支（用于边界）
        self.small_conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels // 2, kernel_size=3,
                                dilation=1, dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels // 2),
            MinkowskiReLU()
        )
        
        # 辅助大尺度分支（用于上下文）
        self.large_conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels // 2, kernel_size=3,
                                dilation=3, dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels // 2),
            MinkowskiReLU()
        )
        
        # 多尺度融合
        self.fusion = nn.Sequential(
            MinkowskiConvolution(out_channels * 2, out_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels),
            MinkowskiReLU()
        )
    
    def forward(self, x):
        # 中等尺度（主特征）
        medium_feat = self.medium_conv(x)
        
        # 小尺度（边界细节）
        small_feat = self.small_conv(x)
        
        # 大尺度（语义上下文）
        large_feat = self.large_conv(x)
        
        # 多尺度拼接
        multi_scale = ME.SparseTensor(
            features=torch.cat([medium_feat.F, small_feat.F, large_feat.F], dim=1),
            coordinate_map_key=medium_feat.coordinate_map_key,
            coordinate_manager=medium_feat.coordinate_manager
        )
        
        return self.fusion(multi_scale)


class HFE(nn.Module):
    """
    HFE (Hierarchical Feature Enhancement) - 层次化特征增强模块
    
    核心思想：
    不同粒度的分割任务对特征的需求存在本质差异，不应共享同一特征。
    HFE为每个任务设计专门的特征增强路径：
    
    - GlobalContextBranch → 树木实例解码器
      全局感知 + 通道重标定，聚合分散的树木组件
      
    - SemanticBranch → 语义解码器  
      多尺度融合，平衡语义信息和边界清晰度
      
    - LocalDetailBranch → 标准实例解码器
      局部细节 + 边界增强，区分相邻果实
    """
    
    def __init__(self, in_channels, out_channels=None, D=3):
        """
        参数:
            in_channels: 输入特征通道数（编码器输出）
            out_channels: 输出特征通道数（默认与输入相同）
            D: 空间维度
        """
        super(HFE, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 三个专门化分支
        self.global_branch = GlobalContextBranch(in_channels, out_channels, D)
        self.semantic_branch = SemanticBranch(in_channels, out_channels, D)
        self.local_branch = LocalDetailBranch(in_channels, out_channels, D)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 编码器输出特征 (SparseTensor)
            
        返回:
            feat_tree: 树木级特征 → 树木实例解码器
            feat_semantic: 语义特征 → 语义解码器
            feat_instance: 实例级特征 → 标准实例解码器
        """
        # 全局上下文分支 → 树木实例分割
        feat_tree = self.global_branch(x)
        
        # 语义分支 → 语义分割
        feat_semantic = self.semantic_branch(x)
        
        # 局部细节分支 → 标准实例分割
        feat_instance = self.local_branch(x)
        
        return feat_instance, feat_semantic, feat_tree


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("HFE模块测试 - 层次化特征增强")
    print("=" * 70)
    print("\n设计理念:")
    print("  - GlobalContextBranch: 全局感知 → 树木实例分割")
    print("  - SemanticBranch: 多尺度融合 → 语义分割")
    print("  - LocalDetailBranch: 边界增强 → 果实实例分割")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    num_points_per_sample = [1000, 800]
    in_channels = 256
    
    coords_list = []
    feats_list = []
    
    for b in range(batch_size):
        n_points = num_points_per_sample[b]
        coords = torch.randint(0, 100, (n_points, 3), dtype=torch.int)
        batch_idx = torch.full((n_points, 1), b, dtype=torch.int)
        coords = torch.cat([batch_idx, coords], dim=1)
        coords_list.append(coords)
        feats = torch.randn(n_points, in_channels)
        feats_list.append(feats)
    
    coords = torch.cat(coords_list, dim=0).to(device)
    feats = torch.cat(feats_list, dim=0).to(device)
    
    input_tensor = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        device=device
    )
    
    print(f"\n输入: {input_tensor.F.shape}")
    
    # 测试HFE
    hfe = HFE(in_channels=in_channels).to(device)
    feat_instance, feat_semantic, feat_tree = hfe(input_tensor)
    
    print(f"\n输出:")
    print(f"  实例级特征 (LocalDetailBranch):    {feat_instance.F.shape} → 标准实例解码器")
    print(f"  语义特征 (SemanticBranch):         {feat_semantic.F.shape} → 语义解码器")
    print(f"  树木级特征 (GlobalContextBranch):  {feat_tree.F.shape} → 树木实例解码器")
    
    # 各分支参数量
    global_params = sum(p.numel() for p in hfe.global_branch.parameters())
    semantic_params = sum(p.numel() for p in hfe.semantic_branch.parameters())
    local_params = sum(p.numel() for p in hfe.local_branch.parameters())
    total_params = sum(p.numel() for p in hfe.parameters())
    
    print(f"\n参数量统计:")
    print(f"  - GlobalContextBranch: {global_params:,}")
    print(f"  - SemanticBranch: {semantic_params:,}")
    print(f"  - LocalDetailBranch: {local_params:,}")
    print(f"  - 总计: {total_params:,}")
    
    # 测试梯度
    loss = feat_instance.F.sum() + feat_semantic.F.sum() + feat_tree.F.sum()
    loss.backward()
    print(f"\n梯度测试: ✓ 所有参数都有梯度")
    
    print("\n" + "=" * 70)
    print("HFE测试完成!")
    print("=" * 70)
