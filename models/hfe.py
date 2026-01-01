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
    2. 全局上下文注入：通过全局池化获取场景级信息（可选）
    3. 通道重标定：选择对全局结构敏感的通道（可选）
    
    消融配置：
    - dilation: 膨胀率
    - use_global_pool: 是否使用全局池化
    - channel_reduction: 通道重标定压缩比（None表示禁用）
    """
    
    def __init__(self, in_channels, out_channels, cfg=None, D=3):
        super(GlobalContextBranch, self).__init__()
        
        # 解析配置
        if cfg is None:
            cfg = {}
        self.dilation = cfg.get('dilation', 4)
        self.use_global_pool = cfg.get('use_global_pool', True)
        self.channel_reduction = cfg.get('channel_reduction', 4)
        
        # 大膨胀率卷积 - 扩大感受野
        self.dilated_conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                dilation=self.dilation, dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels),
            MinkowskiReLU()
        )
        
        # 全局上下文提取（可选）
        if self.use_global_pool and self.channel_reduction:
            self.global_pool = MinkowskiGlobalPooling()
            # 通道重标定（SE风格）- 选择全局结构相关通道
            reduction = self.channel_reduction if self.channel_reduction else 4
            self.channel_gate = nn.Sequential(
                nn.Linear(out_channels, out_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // reduction, out_channels),
                nn.Sigmoid()
            )
        else:
            self.global_pool = None
            self.channel_gate = None
        
        # 特征融合
        self.fusion = nn.Sequential(
            MinkowskiConvolution(out_channels, out_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels)
        )
    
    def forward(self, x):
        # 大感受野特征
        feat = self.dilated_conv(x)
        
        # 全局上下文 + 通道重标定（可选）
        if self.global_pool is not None and self.channel_gate is not None:
            global_context = self.global_pool(feat).F  # [B, C]
            channel_weights = self.channel_gate(global_context)  # [B, C]
            
            # 通道重标定：将全局权重广播到每个点
            batch_indices = feat.C[:, 0].long()
            weights_expanded = channel_weights[batch_indices]  # [N, C]
            
            # 应用通道权重
            enhanced_feat = feat.F * weights_expanded
            
            feat = ME.SparseTensor(
                features=enhanced_feat,
                coordinate_map_key=feat.coordinate_map_key,
                coordinate_manager=feat.coordinate_manager
            )
        
        return self.fusion(feat)


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
    3. 梯度增强：强化边界响应（可选）
    
    消融配置：
    - dilation: 膨胀率
    - use_edge_enhance: 是否使用边界增强
    """
    
    def __init__(self, in_channels, out_channels, cfg=None, D=3):
        super(LocalDetailBranch, self).__init__()
        
        # 解析配置
        if cfg is None:
            cfg = {}
        self.dilation = cfg.get('dilation', 1)
        self.use_edge_enhance = cfg.get('use_edge_enhance', True)
        
        # 标准卷积 - 保留局部细节
        self.local_conv = nn.Sequential(
            MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                dilation=self.dilation, dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels),
            MinkowskiReLU()
        )
        
        # 边界增强卷积（可选）- 使用3x3卷积模拟梯度算子
        if self.use_edge_enhance:
            self.edge_conv = nn.Sequential(
                MinkowskiConvolution(out_channels, out_channels, kernel_size=3,
                                    dimension=D, bias=False),
                MinkowskiBatchNorm(out_channels)
            )
            # 特征融合（边界增强时需要2倍通道）
            self.fusion = nn.Sequential(
                MinkowskiConvolution(out_channels * 2, out_channels, kernel_size=1,
                                    dimension=D, bias=False),
                MinkowskiBatchNorm(out_channels),
                MinkowskiReLU()
            )
        else:
            self.edge_conv = None
            # 无边界增强时，只需要简单的1x1卷积
            self.fusion = nn.Sequential(
                MinkowskiConvolution(out_channels, out_channels, kernel_size=1,
                                    dimension=D, bias=False),
                MinkowskiBatchNorm(out_channels),
                MinkowskiReLU()
            )
    
    def forward(self, x):
        # 局部特征
        local_feat = self.local_conv(x)
        
        if self.use_edge_enhance and self.edge_conv is not None:
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
        else:
            # 无边界增强，直接返回融合后的局部特征
            return self.fusion(local_feat)


class SemanticBranch(nn.Module):
    """
    语义分支 - 用于语义分割
    
    设计动机：
    语义分割关注的是"这个点是什么类别"，需要：
    1. 平衡的上下文：既不能太局部（丢失语义），也不能太全局（丢失边界）
    2. 多尺度融合：结合局部特征和上下文信息（可选）
    3. 类别区分度：增强不同类别间的特征差异
    
    实现方式：
    1. 中等膨胀率：平衡感受野大小
    2. 双尺度融合：结合两种尺度的特征（可选）
    
    消融配置：
    - dilations: 膨胀率列表，如[1,2,3]表示多尺度，[2]表示单尺度
    - use_multiscale: 是否使用多尺度融合
    """
    
    def __init__(self, in_channels, out_channels, cfg=None, D=3):
        super(SemanticBranch, self).__init__()
        
        # 解析配置
        if cfg is None:
            cfg = {}
        self.dilations = cfg.get('dilations', [1, 2, 3])
        self.use_multiscale = cfg.get('use_multiscale', True)
        
        if self.use_multiscale and len(self.dilations) >= 3:
            # 完整多尺度模式
            # 中等尺度分支
            self.medium_conv = nn.Sequential(
                MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    dilation=self.dilations[1], dimension=D, bias=False),
                MinkowskiBatchNorm(out_channels),
                MinkowskiReLU()
            )
            
            # 辅助小尺度分支（用于边界）
            self.small_conv = nn.Sequential(
                MinkowskiConvolution(in_channels, out_channels // 2, kernel_size=3,
                                    dilation=self.dilations[0], dimension=D, bias=False),
                MinkowskiBatchNorm(out_channels // 2),
                MinkowskiReLU()
            )
            
            # 辅助大尺度分支（用于上下文）
            self.large_conv = nn.Sequential(
                MinkowskiConvolution(in_channels, out_channels // 2, kernel_size=3,
                                    dilation=self.dilations[2], dimension=D, bias=False),
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
            self._mode = 'multiscale'
        else:
            # 单尺度模式（仅膨胀率差异）
            dilation = self.dilations[0] if len(self.dilations) == 1 else 2
            self.medium_conv = nn.Sequential(
                MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    dilation=dilation, dimension=D, bias=False),
                MinkowskiBatchNorm(out_channels),
                MinkowskiReLU()
            )
            self.fusion = nn.Sequential(
                MinkowskiConvolution(out_channels, out_channels, kernel_size=1,
                                    dimension=D, bias=False),
                MinkowskiBatchNorm(out_channels),
                MinkowskiReLU()
            )
            self._mode = 'single'
    
    def forward(self, x):
        if self._mode == 'multiscale':
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
        else:
            # 单尺度模式
            feat = self.medium_conv(x)
            return self.fusion(feat)


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
    
    轻量化设计：
    为降低计算开销，采用"压缩-处理-恢复"策略：
    1. 共享压缩层将高维特征投影到低维空间
    2. 各分支在低维空间进行专门化处理
    3. 各分支独立恢复到原始维度
    
    消融配置示例:
        hfe:
          enabled: True
          reduction: 4  # 通道压缩比
          global_branch:
            dilation: 4
            use_global_pool: True
            channel_reduction: 4
          semantic_branch:
            dilations: [1, 2, 3]
            use_multiscale: True
          local_branch:
            dilation: 1
            use_edge_enhance: True
    """
    
    def __init__(self, in_channels, out_channels=None, cfg=None, D=3):
        """
        参数:
            in_channels: 输入特征通道数（编码器输出）
            out_channels: 输出特征通道数（默认与输入相同）
            cfg: 配置字典，控制各分支的消融开关
            D: 空间维度
        """
        super(HFE, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 解析配置
        if cfg is None:
            cfg = {}
        
        # 通道压缩比（轻量化设计的关键）
        self.reduction = cfg.get('reduction', 4)
        reduced_channels = max(in_channels // self.reduction, 32)  # 最小32通道
        
        global_cfg = cfg.get('global_branch', {})
        semantic_cfg = cfg.get('semantic_branch', {})
        local_cfg = cfg.get('local_branch', {})
        
        # 共享通道压缩层 - 降低后续分支的计算量
        self.channel_compress = nn.Sequential(
            MinkowskiConvolution(in_channels, reduced_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(reduced_channels),
            MinkowskiReLU()
        )
        
        # 三个专门化分支，在压缩通道空间中处理
        self.global_branch = GlobalContextBranch(reduced_channels, reduced_channels, cfg=global_cfg, D=D)
        self.semantic_branch = SemanticBranch(reduced_channels, reduced_channels, cfg=semantic_cfg, D=D)
        self.local_branch = LocalDetailBranch(reduced_channels, reduced_channels, cfg=local_cfg, D=D)
        
        # 各分支独立的通道恢复层
        self.expand_global = nn.Sequential(
            MinkowskiConvolution(reduced_channels, out_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels)
        )
        self.expand_semantic = nn.Sequential(
            MinkowskiConvolution(reduced_channels, out_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels)
        )
        self.expand_local = nn.Sequential(
            MinkowskiConvolution(reduced_channels, out_channels, kernel_size=1,
                                dimension=D, bias=False),
            MinkowskiBatchNorm(out_channels)
        )
    
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
        # 共享通道压缩
        x_compressed = self.channel_compress(x)
        
        # 全局上下文分支 → 树木实例分割（在压缩空间处理后恢复）
        feat_tree = self.expand_global(self.global_branch(x_compressed))
        
        # 语义分支 → 语义分割
        feat_semantic = self.expand_semantic(self.semantic_branch(x_compressed))
        
        # 局部细节分支 → 标准实例分割
        feat_instance = self.expand_local(self.local_branch(x_compressed))
        
        return feat_instance, feat_semantic, feat_tree


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import torch
    
    print("=" * 70)
    print("HFE模块参数量分析 - 不同消融配置对比")
    print("=" * 70)
    
    in_channels = 256
    out_channels = 256
    
    # ==================== 配置定义 ====================
    configs = {
        '完整设计 (Full)': {
            'global_branch': {'dilation': 4, 'use_global_pool': True, 'channel_reduction': 4},
            'semantic_branch': {'dilations': [1, 2, 3], 'use_multiscale': True},
            'local_branch': {'dilation': 1, 'use_edge_enhance': True}
        },
        '仅膨胀率差异 (Dilation Only)': {
            'global_branch': {'dilation': 4, 'use_global_pool': False, 'channel_reduction': None},
            'semantic_branch': {'dilations': [2], 'use_multiscale': False},
            'local_branch': {'dilation': 1, 'use_edge_enhance': False}
        },
        '无全局池化': {
            'global_branch': {'dilation': 4, 'use_global_pool': False, 'channel_reduction': None},
            'semantic_branch': {'dilations': [1, 2, 3], 'use_multiscale': True},
            'local_branch': {'dilation': 1, 'use_edge_enhance': True}
        },
        '无多尺度融合': {
            'global_branch': {'dilation': 4, 'use_global_pool': True, 'channel_reduction': 4},
            'semantic_branch': {'dilations': [2], 'use_multiscale': False},
            'local_branch': {'dilation': 1, 'use_edge_enhance': True}
        },
        '无边界增强': {
            'global_branch': {'dilation': 4, 'use_global_pool': True, 'channel_reduction': 4},
            'semantic_branch': {'dilations': [1, 2, 3], 'use_multiscale': True},
            'local_branch': {'dilation': 1, 'use_edge_enhance': False}
        },
    }
    
    results = {}
    
    for config_name, cfg in configs.items():
        print(f"\n{'='*70}")
        print(f"配置: {config_name}")
        print(f"{'='*70}")
        
        # 创建HFE模块
        hfe = HFE(in_channels, out_channels, cfg=cfg)
        
        # 打印配置详情
        print(f"\n配置详情:")
        print(f"  GlobalBranch:")
        print(f"    - dilation: {cfg['global_branch'].get('dilation', 4)}")
        print(f"    - use_global_pool: {cfg['global_branch'].get('use_global_pool', True)}")
        print(f"    - channel_reduction: {cfg['global_branch'].get('channel_reduction', 4)}")
        print(f"  SemanticBranch:")
        print(f"    - dilations: {cfg['semantic_branch'].get('dilations', [1,2,3])}")
        print(f"    - use_multiscale: {cfg['semantic_branch'].get('use_multiscale', True)}")
        print(f"  LocalBranch:")
        print(f"    - dilation: {cfg['local_branch'].get('dilation', 1)}")
        print(f"    - use_edge_enhance: {cfg['local_branch'].get('use_edge_enhance', True)}")
        
        # 统计各分支参数量
        global_params = sum(p.numel() for p in hfe.global_branch.parameters())
        semantic_params = sum(p.numel() for p in hfe.semantic_branch.parameters())
        local_params = sum(p.numel() for p in hfe.local_branch.parameters())
        total_params = sum(p.numel() for p in hfe.parameters())
        
        print(f"\n参数量统计:")
        print(f"  - GlobalContextBranch:  {global_params:>10,}")
        print(f"  - SemanticBranch:       {semantic_params:>10,}")
        print(f"  - LocalDetailBranch:    {local_params:>10,}")
        print(f"  - 总计:                 {total_params:>10,}")
        
        results[config_name] = {
            'global': global_params,
            'semantic': semantic_params,
            'local': local_params,
            'total': total_params
        }
        
        # 打印模型结构
        print(f"\n模块结构:")
        for name, module in hfe.named_modules():
            if name and '.' not in name:  # 只打印一级子模块
                print(f"  {name}:")
                for sub_name, sub_module in module.named_children():
                    sub_params = sum(p.numel() for p in sub_module.parameters())
                    print(f"    └─ {sub_name}: {sub_params:,} params")
    
    # ==================== 对比总结 ====================
    print("\n" + "=" * 70)
    print("参数量对比总结")
    print("=" * 70)
    print(f"\n{'配置':<30} {'Global':>12} {'Semantic':>12} {'Local':>12} {'Total':>12}")
    print("-" * 78)
    
    baseline_total = results['完整设计 (Full)']['total']
    for config_name, params in results.items():
        diff = params['total'] - baseline_total
        diff_str = f"({diff:+,})" if diff != 0 else ""
        print(f"{config_name:<30} {params['global']:>12,} {params['semantic']:>12,} {params['local']:>12,} {params['total']:>12,} {diff_str}")
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
