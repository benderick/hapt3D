# HAPT3D 实验配置系统

## 概述

本配置系统允许通过YAML配置文件灵活切换HFE、CDAG、HCL等模块，方便进行消融实验和参数调优。

## 目录结构

```
config/
├── config_ours.yaml               # 本文完整方法 (HFE + CDAG + HCL)
├── config_baseline.yaml           # 基线方法 (纯MinkUNet14A)
├── config_ablation_hfe.yaml       # 模块消融: 仅HFE
├── config_ablation_cdag.yaml      # 模块消融: 仅CDAG
├── config_ablation_hcl.yaml       # 模块消融: 仅HCL
├── config_ablation_hfe_cdag.yaml  # 模块消融: HFE + CDAG
│
├── ablation_cdag/                 # CDAG内部消融 (Tab. ablation_cdag)
│   ├── config_cdag_spatial.yaml                    # 仅空间注意力
│   ├── config_cdag_spatial_channel.yaml            # 空间+通道
│   ├── config_cdag_spatial_channel_multiscale.yaml # 空间+通道+多尺度
│   └── config_cdag_full.yaml                       # 完整四维度
│
├── ablation_hfe/                  # HFE内部消融 (Tab. ablation_hfe)
│   ├── config_hfe_dilation_only.yaml  # 仅膨胀率差异
│   └── config_hfe_full.yaml           # 完整专门化设计
│
├── ablation_cdag_pos/             # CDAG位置消融 (Tab. ablation_cdag_pos)
│   ├── config_cdag_pos_sem.yaml       # 仅语义解码器
│   ├── config_cdag_pos_tree.yaml      # 仅树木解码器
│   ├── config_cdag_pos_inst.yaml      # 仅实例解码器
│   └── config_cdag_pos_all.yaml       # 所有解码器
│
└── ablation_hcl/                  # HCL权重消融 (Tab. ablation_hcl)
    ├── config_hcl_lambda_0.yaml       # λ=0
    ├── config_hcl_lambda_0.05.yaml    # λ=0.05
    ├── config_hcl_lambda_0.1.yaml     # λ=0.1 (推荐)
    ├── config_hcl_lambda_0.2.yaml     # λ=0.2
    └── config_hcl_lambda_0.5.yaml     # λ=0.5
```

## 配置文件结构

```yaml
# === 实验元信息 ===
experiment:
  id: "ours_full"              # 实验唯一标识
  description: "描述"          # 实验描述
  seed: 42                     # 随机种子

# === 网络配置 ===
network:
  backbone: "MinkUNet14A"      # 骨干网络
  hfe:
    enabled: True              # 是否启用HFE
    # ... HFE详细配置
  cdag:
    enabled: True              # 是否启用CDAG
    # ... CDAG详细配置

# === 损失函数配置 ===
loss:
  hcl:
    enabled: True              # 是否启用HCL
    weight: 0.1                # HCL权重

# === 训练配置 ===
train:
  max_epoch: 500
  lr: 0.005
  # ...
```

## 使用方法

### 1. 使用训练脚本

```bash
# 必须指定配置文件
python train_v2.py -c config/config_ours.yaml

# 可选：覆盖参数
python train_v2.py -c config/config_baseline.yaml --lr 0.001 --epochs 200

# 调试模式
python train_v2.py -c config/config_ours.yaml --debug
```

### 2. 批量运行消融实验

```bash
# 查看所有实验
python run_ablation.py --list

# 模块级消融 (论文 Tab. ablation_modules)
python run_ablation.py --module all

# CDAG组件消融 (论文 Tab. ablation_cdag)
python run_ablation.py --cdag all

# HFE分支消融 (论文 Tab. ablation_hfe)
python run_ablation.py --hfe all

# CDAG位置消融 (论文 Tab. ablation_cdag_pos)
python run_ablation.py --cdag-pos all

# HCL权重消融 (论文 Tab. ablation_hcl)
python run_ablation.py --hcl all

# 运行全部
python run_ablation.py --all
```

### 3. 在代码中使用配置管理器

```python
from utils.config_manager import load_config, ConfigManager

# 加载配置
cfg = load_config("config/config_ours.yaml")

# 访问配置
print(cfg.hfe_enabled)       # True
print(cfg.cdag_enabled)      # True
print(cfg.hcl_enabled)       # True
print(cfg.learning_rate)     # 0.005

# 点分隔路径访问
print(cfg.get("network.hfe.global_branch.dilation"))  # 4

# 配置验证
if cfg.validate():
    print("配置有效")

# 打印摘要
print(cfg.summary())
```

## 消融实验矩阵

### 1. 模块级消融 (Tab. ablation_modules)

| 配置名称 | HFE | CDAG | HCL | 说明 |
|---------|-----|------|-----|------|
| baseline | ✗ | ✗ | ✗ | 基线方法 |
| hfe_only | ✓ | ✗ | ✗ | 验证HFE贡献 |
| cdag_only | ✗ | ✓ | ✗ | 验证CDAG贡献 |
| hcl_only | ✗ | ✗ | ✓ | 验证HCL贡献 |
| hfe_cdag | ✓ | ✓ | ✗ | HFE+CDAG组合 |
| ours | ✓ | ✓ | ✓ | **本文完整方法** |

### 2. CDAG设计选择消融 (Tab. ablation_cdag)

| 配置名称 | 空间 | 通道 | 多尺度 | 像素 |
|---------|-----|------|-------|------|
| cdag_spatial | ✓ | ✗ | ✗ | ✗ |
| cdag_spatial_channel | ✓ | ✓ | ✗ | ✗ |
| cdag_spatial_channel_multiscale | ✓ | ✓ | ✓ | ✗ |
| cdag_full | ✓ | ✓ | ✓ | ✓ |

### 3. HFE分支设计消融 (Tab. ablation_hfe)

| 配置名称 | 全局池化 | 通道重标定 | 多尺度融合 | 边界增强 |
|---------|---------|-----------|-----------|---------|
| hfe_dilation_only | ✗ | ✗ | ✗ | ✗ |
| hfe_full | ✓ | ✓ | ✓ | ✓ |

### 4. CDAG应用位置消融 (Tab. ablation_cdag_pos)

| 配置名称 | 语义解码器 | 树木解码器 | 实例解码器 |
|---------|-----------|-----------|-----------|
| cdag_pos_sem | ✓ | ✗ | ✗ |
| cdag_pos_tree | ✗ | ✓ | ✗ |
| cdag_pos_inst | ✗ | ✗ | ✓ |
| cdag_pos_all | ✓ | ✓ | ✓ |

### 5. HCL损失权重消融 (Tab. ablation_hcl)

| 配置名称 | λ值 |
|---------|-----|
| hcl_lambda_0 | 0 |
| hcl_lambda_0.05 | 0.05 |
| hcl_lambda_0.1 | **0.1 (推荐)** |
| hcl_lambda_0.2 | 0.2 |
| hcl_lambda_0.5 | 0.5 |

## 关键配置项说明

### HFE配置

```yaml
network:
  hfe:
    enabled: True
    global_branch:           # 全局上下文分支 (树木级)
      dilation: 4           # 膨胀率
      use_global_pool: True # 全局池化
      channel_reduction: 4  # 通道压缩比
    semantic_branch:         # 语义分支
      dilations: [1, 2, 3]  # 多尺度膨胀率
    local_branch:            # 局部细节分支 (实例级)
      dilation: 1
      use_edge_enhance: True
```

### CDAG配置

```yaml
network:
  cdag:
    enabled: True
    spatial_gate:            # 空间注意力门控
      enabled: True
    channel_attention:       # 双池化通道注意力
      enabled: True
      reduction: 4
    multiscale_attention:    # 多尺度空间注意力
      enabled: True
      dilations: [1, 2, 3]
    pixel_attention:         # 像素级注意力
      enabled: True
    apply_to:                # 应用位置
      semantic_decoder: True
      tree_decoder: True
      instance_decoder: True
```

### HCL配置

```yaml
loss:
  hcl:
    enabled: True
    weight: 0.1              # λ_HCL
    version: "v1"            # v1:基础版, v2:增强版
    point_wise_alpha: 0.1    # v2逐点约束权重
```

## 实验结果路径

训练完成后，结果保存在:

```
experiments/
├── baseline/
│   └── version_0/
│       ├── checkpoints/     # 模型检查点
│       ├── config.yaml      # 使用的配置副本
│       └── events.*         # TensorBoard日志
├── hfe_only/
│   └── ...
├── ours_full/
│   └── ...
└── ablation_report.txt      # 消融实验报告
```

## 注意事项

1. **配置继承**: 新配置只需覆盖需要修改的项，其他使用默认值
2. **随机种子**: 建议固定seed确保实验可重复性
3. **GPU内存**: batch_size=1适合大多数情况，根据显存调整
4. **早停**: 默认启用，patience=50，可在配置中调整


**************************

# HAPT3D 模块集成指南

## 架构概览

本项目实现了论文《层次感知增强的果园三维全景分割方法》的三大创新模块：

```
┌────────────────────────────────────────────────────────────────────┐
│                         HAPT3D 完整架构                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  输入点云 [N, 3+C]                                                  │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    MinkUNet 编码器                            │  │
│  │   conv0 → block1 → block2 → block3 → block4                 │  │
│  │         (out_p1)  (out_b1p2) (out_b2p4) (out_b3p8)           │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                    │
│                               ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           【创新1】HFE - 层次特征增强模块                      │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  feat_encoder ───────────────────────────────────────▶ │  │  │
│  │  │       │                                                │  │  │
│  │  │       ├─▶ GlobalContextBranch ──▶ feat_tree           │  │  │
│  │  │       │   (d=4, 全局池化, 通道注意力)                    │  │  │
│  │  │       │                                                │  │  │
│  │  │       ├─▶ SemanticBranch ──▶ feat_semantic             │  │  │
│  │  │       │   (多尺度膨胀 d=1,2,3)                          │  │  │
│  │  │       │                                                │  │  │
│  │  │       └─▶ LocalDetailBranch ──▶ feat_instance          │  │  │
│  │  │           (d=1, 边界增强)                               │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                               │                                    │
│       ┌───────────────────────┼───────────────────────┐            │
│       │                       │                       │            │
│       ▼                       ▼                       ▼            │
│  ┌────────────┐         ┌────────────┐         ┌────────────┐      │
│  │ 语义解码器  │         │ 树木解码器  │         │ 实例解码器  │      │
│  │ (feat_sem) │         │(feat_tree) │         │(feat_inst) │      │
│  └─────┬──────┘         └─────┬──────┘         └─────┬──────┘      │
│        │                      │                      │             │
│        │【创新2】CDAG        │【创新2】CDAG        │【创新2】CDAG│
│        │ 门控跳跃连接         │ 门控跳跃连接         │ 门控跳跃连接 │
│        │                      │                      │             │
│        ▼                      ▼                      ▼             │
│  ┌────────────┐         ┌────────────┐         ┌────────────┐      │
│  │ 语义分割头  │         │ 树木偏移头  │         │ 实例偏移头  │      │
│  │  [N, C]    │         │  [N, 3]    │         │  [N, 3]    │      │
│  └─────┬──────┘         └─────┬──────┘         └─────┬──────┘      │
│        │                      │                      │             │
│        ▼                      ▼                      ▼             │
│  ┌────────────┐         ┌────────────┐         ┌────────────┐      │
│  │  L_sem     │         │  L_tree    │         │  L_inst    │      │
│  │(CrossEnt)  │         │ (Lovasz)   │         │ (Lovasz)   │      │
│  └─────┬──────┘         └─────┬──────┘         └─────┬──────┘      │
│        │                      │                      │             │
│        └──────────────────────┼──────────────────────┘             │
│                               │                                    │
│                               ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           【创新3】HCL - 层次一致性损失                        │  │
│  │                                                              │  │
│  │   L_HCL = mean_k( || mean(e_inst_k) - mean(e_tree_k) ||^2 ) │  │
│  │                                                              │  │
│  │   约束: 同一棵树的实例预测中心 ≈ 树木预测中心                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                               │                                    │
│                               ▼                                    │
│                      L_total = L_sem + L_inst + L_tree + λ·L_HCL  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
models/
├── hfe.py              # HFE模块实现
├── cdag.py             # CDAG模块实现  
├── minkunet_ours.py    # 集成HFE+CDAG的MinkUNet骨干
├── hapt3d_ours.py      # 集成HCL的LightningModule
└── __init__.py         # 模块导出

utils/
├── hcl_loss.py         # HCL损失实现
├── config_manager.py   # 配置管理器
└── __init__.py         # 工具导出

config/
├── config_ours.yaml    # 完整方法配置
└── ablation/           # 消融实验配置
    ├── config_baseline.yaml
    ├── config_hfe_only.yaml
    ├── config_cdag_only.yaml
    └── ...
```

## 使用方法

### 1. 训练完整方法

```bash
python train_ours.py -c config/config_ours.yaml
```

### 2. 运行消融实验

```bash
# 基线方法 (无HFE/CDAG/HCL)
python train_ours.py -c config/ablation/config_baseline.yaml

# 仅HFE
python train_ours.py -c config/ablation/config_hfe_only.yaml

# 仅CDAG  
python train_ours.py -c config/ablation/config_cdag_only.yaml

# HFE + CDAG (无HCL)
python train_ours.py -c config/ablation/config_hfe_cdag.yaml
```

### 3. 配置选项说明

```yaml
network:
  # HFE模块配置
  hfe:
    enabled: True/False              # 是否启用HFE
    global_branch:                   # 全局上下文分支配置
      dilation: 4
    semantic_branch:                 # 语义分支配置
      dilations: [1, 2, 3]
    local_branch:                    # 局部细节分支配置
      use_edge_enhance: True
  
  # CDAG模块配置
  cdag:
    enabled: True/False              # 是否启用CDAG
    use_spatial_attention: True      # 空间注意力门控
    use_channel_attention: True      # 双池化通道注意力
    use_multiscale_attention: True   # 多尺度空间注意力
    use_pixel_attention: True        # 像素级注意力
    apply_to:                        # 应用位置
      semantic_decoder: True
      tree_decoder: True
      instance_decoder: True

loss:
  # HCL损失配置
  hcl:
    enabled: True/False              # 是否启用HCL
    weight: 0.1                      # λ_hcl 权重
```

## 模块接口

### HFE模块

```python
from models import HFE

hfe = HFE(in_channels=256, out_channels=256, cfg=hfe_cfg)
feat_instance, feat_semantic, feat_tree = hfe(encoder_output)
```

### CDAG模块

```python
from models import CDAG

cdag = CDAG(F_g=256, F_l=128, cfg=cdag_cfg)
gated_features = cdag(decoder_features, encoder_skip_features)
```

### HCL损失

```python
from utils import HierarchicalConsistencyLoss

hcl = HierarchicalConsistencyLoss()
loss = hcl(coords, offset_inst, offset_tree, tree_labels, valid_mask)
```

## 消融实验对应表

| 配置文件 | HFE | CDAG | HCL | 说明 |
|---------|-----|------|-----|------|
| config_baseline.yaml | ✗ | ✗ | ✗ | 基线方法 |
| config_hfe_only.yaml | ✓ | ✗ | ✗ | 仅HFE |
| config_cdag_only.yaml | ✗ | ✓ | ✗ | 仅CDAG |
| config_hcl_only.yaml | ✗ | ✗ | ✓ | 仅HCL |
| config_hfe_cdag.yaml | ✓ | ✓ | ✗ | HFE+CDAG |
| config_ours.yaml | ✓ | ✓ | ✓ | 完整方法 |

