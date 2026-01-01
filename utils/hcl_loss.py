# -*- coding: utf-8 -*-
"""
HCL (Hierarchical Consistency Loss) - 层次一致性损失

=== 设计动机 ===

在层次化全景分割任务中，存在天然的父子实例关系：
- 一棵树（tree）包含一个树干（trunk）和若干果实（fruit）
- 果实在空间上应该聚集在其所属树木的中心区域附近

然而，现有方法的损失函数独立优化各个子任务：
- 标准实例分割损失：让每个果实的偏移向量指向该果实的中心
- 树木实例分割损失：让每个点的偏移向量指向所属树木的中心

问题在于：这两个损失之间没有显式的关联约束，可能导致：
- 同一棵树的果实被预测到不同的树木实例中
- 果实的预测中心偏离其所属树木的预测中心

=== 核心思想 ===

HCL（层次一致性损失）显式约束：
"同一棵树内所有点的标准实例预测中心的均值，应与该树的树木实例预测中心的均值一致"

数学形式：
对于每棵树 T_k，计算：
- 标准实例预测中心均值: mean(p_i + o_inst_i) for i in T_k
- 树木实例预测中心均值: mean(p_i + o_tree_i) for i in T_k

L_HCL = mean_over_trees( || 标准实例中心均值 - 树木实例中心均值 ||^2 )

作者: [论文作者]
日期: 2025
"""

import torch
import torch.nn as nn


class HierarchicalConsistencyLoss(nn.Module):
    """
    层次一致性损失 (Hierarchical Consistency Loss, HCL)
    
    约束同一棵树内的子实例（果实、树干）预测中心与父实例（树木）预测中心保持一致。
    
    输入:
        coords: 点云坐标 [N, 3]
        offset_inst: 标准实例解码器输出的偏移向量 [N, 3]
        offset_tree: 树木实例解码器输出的偏移向量 [N, 3]
        tree_labels: 每个点所属的树木实例ID [N]
        valid_mask: 有效点掩码（仅things类参与计算）[N]，可选
    
    输出:
        loss: 层次一致性损失标量
    """
    
    def __init__(self, reduction='mean'):
        """
        参数:
            reduction: 损失聚合方式，'mean'或'sum'
        """
        super(HierarchicalConsistencyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, coords, offset_inst, offset_tree, tree_labels, valid_mask=None):
        """
        前向传播计算HCL损失
        
        参数:
            coords: 点云坐标 [N, 3] 或 [B, N, 3]
            offset_inst: 标准实例偏移向量 [N, 3] 或 [B, N, 3]
            offset_tree: 树木实例偏移向量 [N, 3] 或 [B, N, 3]
            tree_labels: 树木实例标签 [N] 或 [B, N]，-1或0表示无效（stuff类）
            valid_mask: 有效点掩码 [N] 或 [B, N]，可选
            
        返回:
            loss: HCL损失值
        """
        # 处理批次维度
        if coords.dim() == 3:
            # 批处理模式 [B, N, 3]
            batch_size = coords.shape[0]
            total_loss = 0.0
            valid_trees = 0
            
            for b in range(batch_size):
                b_coords = coords[b]  # [N, 3]
                b_offset_inst = offset_inst[b]  # [N, 3]
                b_offset_tree = offset_tree[b]  # [N, 3]
                b_tree_labels = tree_labels[b]  # [N]
                b_valid_mask = valid_mask[b] if valid_mask is not None else None
                
                loss, n_trees = self._compute_single_sample(
                    b_coords, b_offset_inst, b_offset_tree, b_tree_labels, b_valid_mask
                )
                total_loss += loss
                valid_trees += n_trees
            
            if valid_trees == 0:
                return torch.tensor(0.0, device=coords.device, requires_grad=True)
            
            if self.reduction == 'mean':
                return total_loss / valid_trees
            else:
                return total_loss
        else:
            # 单样本模式 [N, 3]
            loss, n_trees = self._compute_single_sample(
                coords, offset_inst, offset_tree, tree_labels, valid_mask
            )
            
            if n_trees == 0:
                return torch.tensor(0.0, device=coords.device, requires_grad=True)
            
            if self.reduction == 'mean':
                return loss / n_trees
            else:
                return loss
    
    def _compute_single_sample(self, coords, offset_inst, offset_tree, tree_labels, valid_mask=None):
        """
        计算单个样本的HCL损失
        
        返回:
            loss: 损失值（未归一化）
            n_trees: 有效树木数量
        """
        device = coords.device
        
        # 应用有效掩码（仅things类参与）
        if valid_mask is not None:
            valid_idx = valid_mask.bool()
            coords = coords[valid_idx]
            offset_inst = offset_inst[valid_idx]
            offset_tree = offset_tree[valid_idx]
            tree_labels = tree_labels[valid_idx]
        
        # 过滤无效标签（-1或0通常表示无效/stuff类）
        valid_tree_mask = tree_labels > 0
        if not valid_tree_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        
        coords = coords[valid_tree_mask]
        offset_inst = offset_inst[valid_tree_mask]
        offset_tree = offset_tree[valid_tree_mask]
        tree_labels = tree_labels[valid_tree_mask]
        
        # 计算预测中心
        # e_inst = p + o_inst: 标准实例预测中心
        # e_tree = p + o_tree: 树木实例预测中心
        center_inst = coords + offset_inst  # [N', 3]
        center_tree = coords + offset_tree  # [N', 3]
        
        # 获取所有唯一的树木实例ID
        unique_trees = torch.unique(tree_labels)
        n_trees = len(unique_trees)
        
        if n_trees == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        
        # 对每棵树计算一致性损失
        total_loss = torch.tensor(0.0, device=device)
        
        for tree_id in unique_trees:
            # 获取该树的所有点
            tree_mask = (tree_labels == tree_id)
            n_points = tree_mask.sum()
            
            if n_points < 2:
                # 点数太少，跳过
                continue
            
            # 计算该树的标准实例预测中心均值
            mean_center_inst = center_inst[tree_mask].mean(dim=0)  # [3]
            
            # 计算该树的树木实例预测中心均值
            mean_center_tree = center_tree[tree_mask].mean(dim=0)  # [3]
            
            # 计算L2距离的平方
            loss = torch.sum((mean_center_inst - mean_center_tree) ** 2)
            total_loss = total_loss + loss
        
        return total_loss, n_trees


class HCLv2(nn.Module):
    """
    HCL的增强版本 - 逐点一致性约束
    
    除了约束每棵树的中心均值一致外，还约束每个点的两种预测中心之间的差异。
    
    L_HCL_v2 = L_tree_mean + α * L_point_wise
    
    其中:
    - L_tree_mean: 原始HCL，约束树级中心均值一致
    - L_point_wise: 逐点约束，每个点的inst中心应靠近同树其他点的tree中心均值
    """
    
    def __init__(self, alpha=0.1, reduction='mean'):
        """
        参数:
            alpha: 逐点约束的权重
            reduction: 损失聚合方式
        """
        super(HCLv2, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.base_hcl = HierarchicalConsistencyLoss(reduction=reduction)
    
    def forward(self, coords, offset_inst, offset_tree, tree_labels, valid_mask=None):
        """
        前向传播
        """
        # 基础HCL损失
        loss_tree_mean = self.base_hcl(coords, offset_inst, offset_tree, tree_labels, valid_mask)
        
        # 如果alpha为0，只返回基础损失
        if self.alpha == 0:
            return loss_tree_mean
        
        # 逐点一致性损失（简化版：直接约束两种预测中心的差异）
        loss_point_wise = self._compute_point_wise_loss(
            coords, offset_inst, offset_tree, tree_labels, valid_mask
        )
        
        return loss_tree_mean + self.alpha * loss_point_wise
    
    def _compute_point_wise_loss(self, coords, offset_inst, offset_tree, tree_labels, valid_mask=None):
        """
        计算逐点一致性损失
        
        思想：同一棵树内的点，其inst预测中心和tree预测中心不应相差太远
        """
        device = coords.device
        
        # 处理批次维度
        if coords.dim() == 3:
            coords = coords.reshape(-1, 3)
            offset_inst = offset_inst.reshape(-1, 3)
            offset_tree = offset_tree.reshape(-1, 3)
            tree_labels = tree_labels.reshape(-1)
            if valid_mask is not None:
                valid_mask = valid_mask.reshape(-1)
        
        # 应用有效掩码
        if valid_mask is not None:
            valid_idx = valid_mask.bool()
            coords = coords[valid_idx]
            offset_inst = offset_inst[valid_idx]
            offset_tree = offset_tree[valid_idx]
            tree_labels = tree_labels[valid_idx]
        
        # 过滤无效标签
        valid_tree_mask = tree_labels > 0
        if not valid_tree_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        offset_inst = offset_inst[valid_tree_mask]
        offset_tree = offset_tree[valid_tree_mask]
        
        # 逐点：两种偏移向量的方向应该大致一致（至少不会完全相反）
        # 使用偏移向量的差异作为损失
        diff = offset_inst - offset_tree  # [N', 3]
        loss = torch.mean(torch.sum(diff ** 2, dim=1))
        
        return loss


# ==================== 便捷函数 ====================

def hcl_loss(coords, offset_inst, offset_tree, tree_labels, valid_mask=None, reduction='mean'):
    """
    计算HCL损失的便捷函数
    
    参数:
        coords: 点云坐标 [N, 3]
        offset_inst: 标准实例偏移向量 [N, 3]
        offset_tree: 树木实例偏移向量 [N, 3]
        tree_labels: 树木实例标签 [N]
        valid_mask: 有效点掩码 [N]，可选
        reduction: 聚合方式
        
    返回:
        loss: HCL损失值
    """
    criterion = HierarchicalConsistencyLoss(reduction=reduction)
    return criterion(coords, offset_inst, offset_tree, tree_labels, valid_mask)


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("HCL损失测试 - 层次一致性损失")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 模拟数据
    # 场景：2棵树，每棵树有1个树干+若干果实
    N = 1000  # 总点数
    
    # 点云坐标
    coords = torch.randn(N, 3, device=device)
    
    # 树木标签：0表示stuff（地面等），1和2表示两棵树
    tree_labels = torch.zeros(N, dtype=torch.long, device=device)
    tree_labels[0:400] = 1    # 第一棵树
    tree_labels[400:700] = 2  # 第二棵树
    # 700:1000 是stuff类，标签为0
    
    # 有效掩码：只有things类（树干、果实）参与计算
    valid_mask = tree_labels > 0
    
    # 模拟偏移向量
    # 理想情况：同一棵树的inst和tree偏移应该指向相似的方向
    
    # 情况1：一致的预测（损失应该很小）
    print("\n" + "-" * 50)
    print("测试1：一致的偏移预测")
    print("-" * 50)
    
    # 树1的中心
    tree1_center = torch.tensor([1.0, 1.0, 1.0], device=device)
    # 树2的中心
    tree2_center = torch.tensor([-1.0, -1.0, -1.0], device=device)
    
    offset_inst = torch.zeros(N, 3, device=device)
    offset_tree = torch.zeros(N, 3, device=device)
    
    # 让所有点都指向各自树的中心
    for i in range(N):
        if tree_labels[i] == 1:
            offset_inst[i] = tree1_center - coords[i]
            offset_tree[i] = tree1_center - coords[i]
        elif tree_labels[i] == 2:
            offset_inst[i] = tree2_center - coords[i]
            offset_tree[i] = tree2_center - coords[i]
    
    hcl = HierarchicalConsistencyLoss()
    loss1 = hcl(coords, offset_inst, offset_tree, tree_labels, valid_mask)
    print(f"损失值: {loss1.item():.6f} (应接近0)")
    
    # 情况2：不一致的预测（损失应该较大）
    print("\n" + "-" * 50)
    print("测试2：不一致的偏移预测")
    print("-" * 50)
    
    # inst偏移指向树1中心，tree偏移指向树2中心
    offset_inst_bad = torch.zeros(N, 3, device=device)
    offset_tree_bad = torch.zeros(N, 3, device=device)
    
    for i in range(N):
        if tree_labels[i] == 1:
            offset_inst_bad[i] = tree1_center - coords[i]
            offset_tree_bad[i] = tree2_center - coords[i]  # 错误：指向不同中心
        elif tree_labels[i] == 2:
            offset_inst_bad[i] = tree2_center - coords[i]
            offset_tree_bad[i] = tree1_center - coords[i]  # 错误：指向不同中心
    
    loss2 = hcl(coords, offset_inst_bad, offset_tree_bad, tree_labels, valid_mask)
    print(f"损失值: {loss2.item():.6f} (应远大于测试1)")
    
    # 情况3：测试梯度
    print("\n" + "-" * 50)
    print("测试3：梯度反向传播")
    print("-" * 50)
    
    offset_inst_grad = torch.randn(N, 3, device=device, requires_grad=True)
    offset_tree_grad = torch.randn(N, 3, device=device, requires_grad=True)
    
    loss3 = hcl(coords, offset_inst_grad, offset_tree_grad, tree_labels, valid_mask)
    loss3.backward()
    
    print(f"损失值: {loss3.item():.6f}")
    print(f"offset_inst梯度范数: {offset_inst_grad.grad.norm().item():.6f}")
    print(f"offset_tree梯度范数: {offset_tree_grad.grad.norm().item():.6f}")
    
    # 情况4：测试HCLv2
    print("\n" + "-" * 50)
    print("测试4：HCLv2增强版本")
    print("-" * 50)
    
    hcl_v2 = HCLv2(alpha=0.1)
    offset_inst_v2 = torch.randn(N, 3, device=device, requires_grad=True)
    offset_tree_v2 = torch.randn(N, 3, device=device, requires_grad=True)
    
    loss4 = hcl_v2(coords, offset_inst_v2, offset_tree_v2, tree_labels, valid_mask)
    loss4.backward()
    
    print(f"HCLv2损失值: {loss4.item():.6f}")
    print(f"offset_inst梯度范数: {offset_inst_v2.grad.norm().item():.6f}")
    print(f"offset_tree梯度范数: {offset_tree_v2.grad.norm().item():.6f}")
    
    # 情况5：批处理模式
    print("\n" + "-" * 50)
    print("测试5：批处理模式")
    print("-" * 50)
    
    batch_size = 4
    coords_batch = torch.randn(batch_size, N, 3, device=device)
    offset_inst_batch = torch.randn(batch_size, N, 3, device=device, requires_grad=True)
    offset_tree_batch = torch.randn(batch_size, N, 3, device=device, requires_grad=True)
    tree_labels_batch = tree_labels.unsqueeze(0).expand(batch_size, -1)
    valid_mask_batch = valid_mask.unsqueeze(0).expand(batch_size, -1)
    
    loss5 = hcl(coords_batch, offset_inst_batch, offset_tree_batch, tree_labels_batch, valid_mask_batch)
    loss5.backward()
    
    print(f"批处理损失值: {loss5.item():.6f}")
    print(f"批处理梯度形状: {offset_inst_batch.grad.shape}")
    
    print("\n" + "=" * 70)
    print("HCL测试完成!")
    print("=" * 70)
