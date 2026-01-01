# 🚀 快速使用指南

## 本文方法 (HFE + CDAG + HCL)

### 文件结构

```
关键文件:
├── train_v2.py              # 训练脚本 (使用本文方法)
├── run_ablation.py          # 消融实验统一脚本
├── config/
│   ├── config_ours.yaml     # 完整方法配置 (HFE+CDAG+HCL)
│   ├── config_baseline.yaml # 基线配置
│   └── config_ablation_*.yaml  # 消融实验配置
├── models/
│   ├── hapt3d_ours.py       # 训练模块 (含HCL)
│   ├── minkunet_ours.py     # 网络模型 (含HFE+CDAG)
│   ├── hfe.py               # HFE模块
│   └── cdag.py              # CDAG模块
└── utils/
    ├── hcl_loss.py          # HCL损失函数
    └── config_manager.py    # 配置管理器
```

### 训练命令

```bash
# 1. 训练完整方法 (本文方法)
python train_v2.py -c config/config_ours.yaml

# 2. 训练基线 (用于对比)
python train_v2.py -c config/config_baseline.yaml

# 3. 覆盖配置项
python train_v2.py -c config/config_ours.yaml --lr 0.001 --epochs 200 --gpus 2

# 4. 调试模式 (单epoch)
python train_v2.py -c config/config_ours.yaml --debug
```

### 消融实验

```bash
# 查看所有可用实验
python run_ablation.py --list

# === 模块级消融 (Tab. ablation_modules) ===
python run_ablation.py --module all          # 运行所有
python run_ablation.py --module baseline     # 仅基线
python run_ablation.py --module ours         # 仅完整方法

# === CDAG组件消融 (Tab. ablation_cdag) ===
python run_ablation.py --cdag all

# === HFE分支消融 (Tab. ablation_hfe) ===
python run_ablation.py --hfe all

# === CDAG位置消融 (Tab. ablation_cdag_pos) ===
python run_ablation.py --cdag-pos all

# === HCL权重消融 (Tab. ablation_hcl) ===
python run_ablation.py --hcl all

# === 运行全部实验 ===
python run_ablation.py --all --skip-existing
```

### 配置说明

`config_ours.yaml` 核心配置:

```yaml
network:
  backbone: "MinkUNet14A"
  
  # HFE模块 - 层次特征增强
  hfe:
    enabled: True
    global_branch:     # 全局上下文分支
      dilation: 4
      use_global_pool: True
    semantic_branch:   # 语义分支 (多尺度膨胀)
      dilations: [1, 2, 3]
    local_branch:      # 局部细节分支
      dilation: 1
      use_edge_enhance: True
  
  # CDAG模块 - 通道双重注意力门控
  cdag:
    enabled: True
    use_spatial_gate: True       # 空间注意力门控
    use_channel_attention: True  # 双池化通道注意力
    use_multiscale: True         # 多尺度空间注意力
    use_pixel_attention: True    # 像素级注意力

loss:
  # HCL损失 - 层次一致性损失
  hcl:
    enabled: True
    weight: 0.1                  # λ = 0.1
    temperature: 0.07            # 对比学习温度
```

### 实验对应表

| 论文表格 | 命令 |
|---------|------|
| Tab. ablation_modules | `--module all` |
| Tab. ablation_cdag | `--cdag all` |
| Tab. ablation_hfe | `--hfe all` |
| Tab. ablation_cdag_pos | `--cdag-pos all` |
| Tab. ablation_hcl | `--hcl all` |

### 注意事项

1. **数据路径**: 确保 `data/hopt3d` 目录存在且包含正确的数据
2. **GPU内存**: 完整方法约需 24GB 显存，可调整 `train.batch_size`
3. **依赖安装**: 
   ```bash
   pip install -r requirements.txt
   # MinkowskiEngine 需要单独安装
   ```

## 三大创新模块

### 1. HFE (层次特征增强)
- 位置: [models/hfe.py](models/hfe.py)
- 功能: 从编码器输出生成三种专门化特征
  - 全局上下文分支 → 语义解码器
  - 语义分支 → 树木解码器
  - 局部细节分支 → 实例解码器

### 2. CDAG (通道双重注意力门控)
- 位置: [models/cdag.py](models/cdag.py)
- 功能: 自适应选择跳跃连接特征
  - 空间注意力门控 (SAG)
  - 双池化通道注意力 (DPCA)
  - 多尺度空间注意力 (MSA)
  - 像素级注意力 (PA)

### 3. HCL (层次一致性损失)
- 位置: [utils/hcl_loss.py](utils/hcl_loss.py)
- 功能: 增强跨任务特征一致性
  - 语义-实例一致性
  - 树木-实例一致性
  - 层次对比学习
