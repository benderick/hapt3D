# HAPT3D 项目代码详解

## 📁 项目结构概览

```
hapt3D/
├── train.py                 # 训练入口脚本
├── test.py                  # 测试入口脚本
├── requirements.txt         # 依赖包列表
├── INSTALL.md              # 安装指南
├── Dockerfile              # Docker配置
├── Makefile                # 构建脚本
├── config/                 # 配置文件目录
│   ├── config.yaml         # 默认配置
│   ├── config_full.yaml    # 完整跳跃连接配置（论文方法）
│   ├── config_standard.yaml # 标准跳跃连接配置
│   ├── config_no_skip.yaml  # 无跳跃连接配置
│   └── config_dec_skip.yaml # 仅解码器跳跃连接配置
├── datasets/               # 数据集处理模块
│   ├── __init__.py
│   ├── dataloader.py       # PyTorch Lightning DataModule
│   ├── dataset.py          # 数据集类定义
│   └── tf.py               # 数据增强变换
├── models/                 # 模型定义模块
│   ├── __init__.py
│   ├── hapt3d.py           # 主模型（LightningModule）
│   ├── minkunet.py         # 标准MinkUNet
│   ├── minkunet_full.py    # 完整跳跃连接MinkUNet（论文核心）
│   ├── minkunet_no_skip.py # 无跳跃连接MinkUNet
│   ├── minkunet_decoder_only.py # 仅解码器跳跃连接
│   └── resnet.py           # ResNet基础模块
└── utils/                  # 工具函数模块
    ├── __init__.py
    ├── evaluation.py       # 评估指标计算
    ├── func.py             # 辅助函数
    ├── lovasz.py           # Lovász损失函数
    └── viz.py              # 可视化工具
```

---

## 🔧 环境配置与安装

### 依赖环境

```bash
# 创建conda虚拟环境
conda create --name hapt3d python=3.9
conda activate hapt3d

# 安装PyTorch（CUDA 11.3版本）
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir

# 安装NumPy和setuptools
pip install numpy==1.24.2
pip install setuptools==60.0

# 安装PyKeOps（高效GPU计算）
pip install pykeops --no-cache-dir

# 安装MinkowskiEngine（稀疏3D卷积库）- 核心依赖
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps

# 安装PyTorch Lightning（训练框架）
pip install pytorch-lightning==1.9.0 --no-deps
pip install fsspec lightning-utilities

# 安装其他依赖
pip install tqdm pyyaml torchmetrics==1.4.1 ipdb
pip install open3d tensorboard hdbscan distinctipy
pip install optuna==3.6.1 optuna-integration
```

### 核心依赖说明

| 依赖库 | 版本 | 作用 |
|--------|------|------|
| **MinkowskiEngine** | 最新 | 稀疏3D卷积核心库，处理点云的稀疏张量表示 |
| **PyTorch Lightning** | 1.9.0 | 训练框架，简化训练/验证/测试流程 |
| **Open3D** | - | 点云I/O和处理 |
| **HDBSCAN** | - | 层次聚类，用于实例分割后处理 |
| **torchmetrics** | 1.4.1 | 评估指标计算（IoU, PQ等） |

---

## ⚙️ 配置文件详解

配置文件位于 `config/` 目录，使用YAML格式。以 `config_full.yaml` 为例：

```yaml
experiment:
  id: "full_1"              # 实验标识符，用于区分不同实验

data_path: "data/hopt3d"    # 数据集路径

network:
  tanh: True                # 是否对偏移输出使用tanh激活（归一化到[-1,1]）
  embeddings_only: False    # 是否只输出嵌入向量
  skip: "full"              # 跳跃连接类型: "standard", "no_skip", "decoder_only", "full"
  name: "MinkUNet14A"       # 网络架构名称

tasks:
  semantic_segmentation:
    n_classes: 6            # 语义类别数：0-void, 1-ground, 2-plant, 3-fruit, 4-trunk, 5-pole

train:
  ignore_idx: 0             # 忽略的标签索引（void类别）
  max_epoch: 450            # 最大训练轮数
  lr: 0.005                 # 学习率
  batch_size: 1             # 批次大小（点云通常设为1）
  n_gpus: 1                 # GPU数量
  workers: 0                # 数据加载线程数
  overfit: False            # 是否过拟合模式（用于调试）
  voxel_resolution: 0.003   # 体素分辨率（3mm），影响点云稀疏化程度

transform:                  # 数据增强参数
  min_scalefactor: 0.8      # 最小缩放因子
  max_scalefactor: 1.2      # 最大缩放因子
  max_rotation_angle_degree_x: 15   # X轴最大旋转角度
  max_rotation_angle_degree_y: 15   # Y轴最大旋转角度
  max_rotation_angle_degree_z: 180  # Z轴最大旋转角度
  max_shear: 0.2            # 最大剪切变换
  min_downsample: 0.6       # 最小下采样比例
  max_downsample: 1.0       # 最大下采样比例
  # 颜色增强参数
  min_contrast: 0.8
  max_contrast: 1.2
  max_brightness: 0.2
  max_hue: 0.15
  max_saturation: 0.15

val:
  min_n_points_fruit: 60    # 果实实例最小点数阈值
  min_n_points_trunk: 250   # 树干实例最小点数阈值
  min_n_points_tree: 1000   # 树木实例最小点数阈值
  pq_from_epoch: 50         # 从第50轮开始计算PQ指标

test:
  dump_metrics: True        # 是否保存测试指标到JSON文件
```

### 跳跃连接变体对比

| 配置 | 跳跃连接方式 | 说明 |
|------|-------------|------|
| `standard` | 编码器→解码器 | 传统UNet方式 |
| `no_skip` | 无跳跃连接 | 消融实验用 |
| `decoder_only` | 前序解码器→后续解码器 | 仅解码器间传递 |
| **`full`** | 编码器+前序解码器→解码器 | **论文提出的方法** |

---

## 🗂️ 数据集模块详解

### 1. 数据集类 (`datasets/dataset.py`)

```python
class HAPT3DDataset(Dataset):
    """
    HAPT3D数据集类，用于加载PLY格式的点云文件
    
    数据格式要求:
    - 点云文件: PLY格式
    - 包含字段: x, y, z (坐标), red, green, blue (颜色), 
                semantic, instance, semantic_h, instance_h (标签)
    """
```

**数据加载流程:**

1. **读取PLY文件** - 使用Open3D读取点云
2. **提取坐标和颜色** - 归一化颜色值到[0,1]
3. **加载标签** - 语义/实例标签及其层次版本
4. **点云归一化** - 将坐标中心化并缩放
5. **数据增强** - 几何变换和颜色增强（仅训练时）

**语义类别定义:**

| ID | 类别 | 英文 |
|----|------|------|
| 0 | 空 | void |
| 1 | 地面 | ground |
| 2 | 植物 | plant |
| 3 | 果实 | fruit |
| 4 | 树干 | trunk |
| 5 | 杆柱 | pole |

**类别划分:**
- **Stuff类别** (背景): ground, plant, pole (IDs: 1, 2, 5)
- **Things类别** (可数物体): fruit, trunk (IDs: 3, 4)

### 2. 数据加载器 (`datasets/dataloader.py`)

```python
class HAPT3DDataModule(LightningDataModule):
    """
    PyTorch Lightning数据模块
    
    功能:
    - 创建训练/验证/测试数据集
    - 配置DataLoader参数
    - 管理数据增强开关
    """
    
    def train_dataloader(self):
        # 训练时启用数据增强
        return DataLoader(
            HAPT3DDataset(path, split='train', transform=True, ...),
            batch_size=1,
            collate_fn=lambda x: x[0]  # 直接返回单个样本
        )
```

### 3. 数据增强 (`datasets/tf.py`)

```python
# 几何增强
def geometricaug(coords, cfg, phase='train'):
    """
    几何数据增强，包括:
    - 随机缩放 (scale)
    - 随机旋转 (rotation around x, y, z axes)
    - 随机剪切 (shear)
    - 随机下采样 (downsampling)
    """

# 颜色增强  
def coloraug(colors, cfg, phase='train'):
    """
    颜色数据增强，包括:
    - 对比度调整 (contrast)
    - 亮度调整 (brightness)
    - 色调调整 (hue)
    - 饱和度调整 (saturation)
    """
```

---

## 🧠 模型架构详解

### 1. 主模型类 (`models/hapt3d.py`)

```python
class HAPT3D(LightningModule):
    """
    HAPT3D主模型，继承自PyTorch Lightning的LightningModule
    
    包含:
    - 网络骨架 (MinkUNet)
    - 损失函数计算
    - 训练/验证/测试步骤
    - 后处理和评估
    """
```

**模型初始化关键部分:**

```python
def __init__(self, cfg):
    # 根据配置选择网络架构
    if cfg['network']['skip'] == "standard":
        from models.minkunet import MinkUNet14A as MinkUNet
    elif cfg['network']['skip'] == "full":
        from models.minkunet_full import MinkUNet14A as MinkUNet  # 论文方法
    elif cfg['network']['skip'] == "no_skip":
        from models.minkunet_no_skip import MinkUNet14A as MinkUNet
    elif cfg['network']['skip'] == "decoder_only":
        from models.minkunet_decoder_only import MinkUNet14A as MinkUNet
    
    # 实例化网络
    self.network = MinkUNet(
        in_channels=3,          # 输入通道数 (RGB)
        out_channels=6,         # 语义类别数
        D=3,                    # 3D空间
        embeddings_only=False,
        use_tanh=True
    )
    
    # 损失函数
    self.ce = nn.CrossEntropyLoss(ignore_index=0)  # 语义分割损失
    self.lovasz = IoULovaszLoss(invert=False)      # 实例分割损失
```

### 2. 前向传播

```python
def forward(self, data):
    """
    前向传播流程:
    1. 体素化点云 → 稀疏张量
    2. 网络推理 → 三个输出
    3. 返回预测结果
    
    输入: data字典，包含坐标、颜色、标签等
    输出: (语义预测, 标准实例偏移, 层次实例偏移)
    """
    # 创建稀疏张量
    sinput = ME.SparseTensor(
        features=data['colors'],           # RGB特征
        coordinates=data['quantized'],     # 体素化坐标
        device=self.device
    )
    
    # 网络前向传播
    soutput, ins1, ins2 = self.network(sinput)
    
    # 返回三个输出:
    # soutput: 语义分割结果 (6类)
    # ins1: 标准实例偏移向量 (3D)
    # ins2: 层次实例偏移向量 (3D)
    return soutput, ins1, ins2
```

### 3. 损失函数

```python
def getLoss(self, data, soutput, ins1, ins2):
    """
    损失函数计算:
    
    总损失 = 语义损失 + 标准实例损失 + 层次实例损失
    
    1. 语义损失: CrossEntropyLoss
       - 6类分类任务
       - 忽略void类别(index=0)
    
    2. 标准实例损失: IoU Lovász Loss
       - 仅对things类别(fruit, trunk)计算
       - 基于偏移向量预测
    
    3. 层次实例损失: IoU Lovász Loss
       - 对tree实例计算
       - 使用层次标签(semantic_h, instance_h)
    """
    
    # 语义分割损失
    sem_loss = self.ce(soutput.F, sem_labels.long())
    
    # 标准实例损失 (fruit + trunk)
    ins1_loss = self.lovasz(
        ins1.F[things_mask],      # 偏移预测
        coords[things_mask],       # 点坐标
        instance[things_mask]      # 实例标签
    )
    
    # 层次实例损失 (tree)
    ins2_loss = self.lovasz(
        ins2.F[things_h_mask],
        coords[things_h_mask],
        instance_h[things_h_mask]
    )
    
    return sem_loss + ins1_loss + ins2_loss, sem_loss, ins1_loss, ins2_loss
```

### 4. MinkUNet架构 (`models/minkunet_full.py`)

**网络结构概览:**

```
输入 → 编码器 → 三个并行解码器 → 三个输出
                ├── 语义解码器 → 语义预测 (6类)
                ├── 层次实例解码器 → 偏移向量 (3D) → 树实例
                └── 标准实例解码器 → 偏移向量 (3D) → 果实/树干实例
```

**编码器结构:**

```python
# 编码器 - 逐步下采样
conv0 → block1 (stride=2)  # 输出: out_b1p2, 步幅2
      → block2 (stride=2)  # 输出: out_b2p4, 步幅4  
      → block3 (stride=2)  # 输出: out_b3p8, 步幅8
      → block4 (stride=2)  # 输出: out_encoder, 步幅16
```

**语义解码器结构:**

```python
# 语义解码器 - 逐步上采样 + 编码器跳跃连接
convtr4 → cat(out_skip_sem1, out_b3p8) → block5  # 步幅8
convtr5 → cat(out_skip_sem2, out_b2p4) → block6  # 步幅4
convtr6 → cat(out_skip_sem3, out_b1p2) → block7  # 步幅2
convtr7 → cat(out_skip_sem4, out_p1)   → block8  # 步幅1
final → 6类预测
```

**层次实例解码器 (ins2) - 论文核心创新:**

```python
# 层次实例解码器 - 编码器 + 语义解码器的跳跃连接
convtr4_ins2 → cat(out_ins2, out_b3p8, out_skip_sem1) → block5_ins2
convtr5_ins2 → cat(out_ins2, out_b2p4, out_skip_sem2) → block6_ins2
convtr6_ins2 → cat(out_ins2, out_b1p2, out_skip_sem3) → block7_ins2
convtr7_ins2 → cat(out_ins2, out_p1, out_skip_sem4)   → block8_ins2
final_ins2 → 3D偏移向量 → tanh激活
```

**标准实例解码器 (ins1) - 论文核心创新:**

```python
# 标准实例解码器 - 编码器 + 层次实例解码器的跳跃连接
convtr4_ins1 → cat(out_ins1, out_b3p8, out_skip_ins1) → block5_ins1
convtr5_ins1 → cat(out_ins1, out_b2p4, out_skip_ins2) → block6_ins1
convtr6_ins1 → cat(out_ins1, out_b1p2, out_skip_ins3) → block7_ins1
convtr7_ins1 → cat(out_ins1, out_p1, out_skip_ins4)   → block8_ins1
final_ins1 → 3D偏移向量 → tanh激活
```

**跳跃连接可视化:**

```
           ┌─────────────────────────────────────────────────────────┐
           │                    编码器特征                            │
           └──────┬──────────────┬───────────────┬───────────────┬───┘
                  │              │               │               │
                  ▼              ▼               ▼               ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                   语义解码器                                 │
        │   block5 ─→ block6 ─→ block7 ─→ block8 ─→ 语义预测         │
        └──────┬──────────────┬───────────────┬───────────────┬───────┘
               │              │               │               │
               ▼              ▼               ▼               ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 层次实例解码器 (ins2)                        │
        │   block5 ─→ block6 ─→ block7 ─→ block8 ─→ 树偏移            │
        └──────┬──────────────┬───────────────┬───────────────┬───────┘
               │              │               │               │
               ▼              ▼               ▼               ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 标准实例解码器 (ins1)                        │
        │   block5 ─→ block6 ─→ block7 ─→ block8 ─→ 果实/树干偏移     │
        └─────────────────────────────────────────────────────────────┘
```

---

## 📊 后处理与评估

### 1. 实例聚类后处理

```python
def post_processing(self, soutput, ins1, ins2, data):
    """
    后处理流程:
    
    1. 获取语义预测 (argmax)
    2. 计算实例中心预测
    3. HDBSCAN聚类生成实例分割
    
    关键参数:
    - min_n_points_fruit: 60 (果实最小点数)
    - min_n_points_trunk: 250 (树干最小点数)
    - min_n_points_tree: 1000 (树木最小点数)
    """
    
    # 语义预测
    sem = soutput.F.argmax(dim=1)
    
    # 实例中心预测 = 点坐标 + 偏移向量
    ins1_centers = coords + ins1.F  # 标准实例中心
    ins2_centers = coords + ins2.F  # 层次实例中心
    
    # HDBSCAN聚类
    for cls_id in [3, 4]:  # fruit, trunk
        mask = (sem == cls_id)
        if mask.sum() > min_points:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_points)
            clusters = clusterer.fit_predict(ins1_centers[mask])
            # 分配实例ID
```

### 2. 评估指标

```python
class Metrics:
    """
    评估指标类，计算:
    
    1. mIoU (Mean Intersection over Union)
       - 语义分割评估
       - 忽略void类别后5类的平均IoU
    
    2. PQ (Panoptic Quality)
       - 标准全景分割评估
       - 分别评估things和stuff类别
       - PQ = SQ × RQ (分割质量 × 识别质量)
    
    3. PQ_h (Hierarchical Panoptic Quality)
       - 层次全景分割评估
       - 评估树级别的实例分割
    """
```

**指标计算:**

| 指标 | 计算方式 | 评估对象 |
|------|---------|---------|
| mIoU | 各类IoU的平均值 | 语义分割 |
| PQ | SQ × RQ | 标准全景分割 |
| PQ_h | SQ × RQ (树级别) | 层次全景分割 |

---

## 🚀 运行指南

### 1. 训练模型

```bash
# 使用默认配置训练
python train.py

# 使用特定配置文件训练
python train.py --config config/config_full.yaml
```

**训练脚本关键参数:**

```python
# train.py 核心代码
trainer = Trainer(
    max_epochs=cfg['train']['max_epoch'],  # 最大训练轮数
    accelerator='gpu',                      # 使用GPU
    devices=cfg['train']['n_gpus'],         # GPU数量
    logger=tb_logger,                       # TensorBoard日志
    callbacks=[                             # 回调函数
        # 多个ModelCheckpoint，监控不同指标
        checkpoint_miou,     # 监控mIoU
        checkpoint_mpq,      # 监控mPQ
        checkpoint_pqh,      # 监控PQ_h
        checkpoint_ins1,     # 监控ins1_loss
        checkpoint_ins2,     # 监控ins2_loss
    ]
)
```

**训练输出:**
- 模型检查点: `lightning_logs/version_X/checkpoints/`
- TensorBoard日志: `lightning_logs/version_X/`
- 最佳模型按不同指标保存

### 2. 测试模型

```bash
# 测试训练好的模型
python test.py --checkpoint path/to/checkpoint.ckpt
```

### 3. 查看训练日志

```bash
# 启动TensorBoard
tensorboard --logdir lightning_logs/
```

### 4. Docker运行

```bash
# 构建Docker镜像
docker-compose build

# 启动容器
docker-compose up
```

---

## 📈 损失函数详解

### 1. 语义分割损失 (CrossEntropyLoss)

```python
# 标准交叉熵损失
ce_loss = CrossEntropyLoss(ignore_index=0)  # 忽略void类别

# 计算
sem_loss = ce_loss(predictions, labels)
```

### 2. 实例分割损失 (IoU Lovász Loss)

```python
class IoULovaszLoss:
    """
    基于Lovász扩展的IoU损失函数
    
    原理:
    1. 从偏移向量计算软掩码
    2. 应用Lovász梯度进行排序
    3. 计算IoU损失
    
    优势:
    - 直接优化IoU指标
    - 对不平衡数据更鲁棒
    """
    
    def forward(self, offsets, coordinates, instance_labels):
        # 计算预测中心
        pred_centers = coordinates + offsets
        
        # 为每个实例计算软掩码
        for instance_id in unique_instances:
            # 获取实例质心
            centroid = mean(pred_centers[mask])
            
            # 计算到质心的距离 → 软掩码
            distances = ||pred_centers - centroid||
            soft_mask = 1 - sigmoid(distances)
            
            # Lovász损失
            loss += lovasz_hinge(soft_mask, ground_truth_mask)
        
        return loss
```

---

## 🔍 关键代码片段解析

### 1. 体素化处理

```python
# 将连续坐标量化为离散体素
voxel_resolution = 0.003  # 3mm分辨率

# 量化坐标
quantized = torch.floor(coords / voxel_resolution).int()

# 创建稀疏张量
sinput = ME.SparseTensor(
    features=colors,
    coordinates=quantized,
    device=device
)
```

### 2. 稀疏卷积操作

```python
# MinkowskiEngine稀疏卷积
conv = ME.MinkowskiConvolution(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=2,
    dimension=3  # 3D空间
)

# 转置卷积（上采样）
convtr = ME.MinkowskiConvolutionTranspose(
    in_channels=64,
    out_channels=32,
    kernel_size=2,
    stride=2,
    dimension=3
)

# 特征拼接
out = ME.cat(feature1, feature2, feature3)
```

### 3. HDBSCAN聚类

```python
import hdbscan

# 创建聚类器
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=60,     # 最小聚类大小
    cluster_selection_epsilon=0.1
)

# 执行聚类
cluster_labels = clusterer.fit_predict(predicted_centers)

# -1 表示噪声点
valid_clusters = cluster_labels[cluster_labels >= 0]
```

---

## 🎯 调参建议

### 1. 学习率调整

```yaml
train:
  lr: 0.005  # 默认值
  # 建议范围: 0.001 - 0.01
  # 如果loss震荡，尝试减小
  # 如果收敛太慢，尝试增大
```

### 2. 体素分辨率

```yaml
train:
  voxel_resolution: 0.003  # 3mm
  # 更小的值 → 更多细节，更大内存消耗
  # 更大的值 → 更快处理，可能丢失细节
```

### 3. 数据增强强度

```yaml
transform:
  # 几何增强
  max_rotation_angle_degree_z: 180  # Z轴旋转范围
  max_shear: 0.2                    # 剪切强度
  
  # 颜色增强
  max_brightness: 0.2               # 亮度变化
  max_hue: 0.15                     # 色调变化
```

### 4. 实例后处理阈值

```yaml
val:
  min_n_points_fruit: 60    # 果实最小点数
  min_n_points_trunk: 250   # 树干最小点数
  min_n_points_tree: 1000   # 树木最小点数
  # 增大 → 更少但更可靠的实例
  # 减小 → 更多但可能有噪声的实例
```

---

## 📝 代码扩展指南

### 1. 添加新的语义类别

```python
# 1. 修改配置文件
tasks:
  semantic_segmentation:
    n_classes: 7  # 增加一类

# 2. 修改评估代码 (utils/evaluation.py)
STUFF_IDS = [1, 2, 5, 6]  # 添加新的stuff类别
# 或
THINGS_IDS = [3, 4, 6]    # 添加新的things类别

# 3. 更新数据集标签
```

### 2. 使用自定义数据集

```python
# 确保PLY文件包含以下字段:
# - x, y, z: 点坐标
# - red, green, blue: RGB颜色 (0-255)
# - semantic: 语义标签
# - instance: 实例ID
# - semantic_h: 层次语义标签
# - instance_h: 层次实例ID

# 数据集目录结构:
# data/your_dataset/
#   ├── train/
#   │   ├── scene1.ply
#   │   └── scene2.ply
#   ├── val/
#   │   └── scene3.ply
#   └── test/
#       └── scene4.ply
```

### 3. 修改网络架构

```python
# 在 models/ 目录下创建新的网络文件
# 继承 MinkUNetBase 类
class CustomMinkUNet(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)  # 自定义层数
    # ...
```

---

## 🐛 常见问题排查

### 1. CUDA内存不足

```bash
# 错误: CUDA out of memory
# 解决方案:
# 1. 增大体素分辨率
voxel_resolution: 0.005  # 从0.003增大到0.005

# 2. 减少数据增强的下采样
min_downsample: 0.8  # 从0.6增大到0.8
```

### 2. MinkowskiEngine安装失败

```bash
# 确保安装了正确版本的依赖
pip install numpy==1.24.2
pip install setuptools==60.0

# 从源码安装
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
```

### 3. 训练loss不下降

```python
# 检查学习率是否合适
lr: 0.001  # 尝试减小

# 检查数据增强是否过强
max_rotation_angle_degree_z: 90  # 减小旋转范围

# 检查体素分辨率
voxel_resolution: 0.005  # 尝试增大
```

---

## 📚 参考资料

- **MinkowskiEngine文档**: https://nvidia.github.io/MinkowskiEngine/
- **PyTorch Lightning文档**: https://lightning.ai/docs/pytorch/stable/
- **HDBSCAN文档**: https://hdbscan.readthedocs.io/
- **Open3D文档**: http://www.open3d.org/docs/

---

*本文档由代码分析自动生成，如有疑问请参考源代码或联系作者。*



# HAPT3D

### Train
Run `python train.py --config config/config_full.yaml`. Remember to change the path to the dataset folder in the config file and in the `train.py` file.

### Testing
Run `python test.py -w <file>`. Remember to change the path to the dataset folder in the config file and in the `test.py` file. If you want to test on the validation set, uncomment lines 41-44 in `test.py`.

### Installation
After struggling a bit to install MinkowskiEngine, the procedure below is the one that worked out on my machine (operations to be done in that specific order):

```
    conda create --name hapt3d python=3.9
    conda activate hapt3d
    pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
    pip install numpy==1.24.2
    pip install setuptools==60.0
    pip install pykeops --no-cache-di
    pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
    pip install pytorch-lightning==1.9.0 --no-deps
    pip install fsspec
    pip install lightning-utilities
    pip install tqdm
    pip install pyyaml
    pip install torchmetrics==1.4.1
    pip install ipdb
    pip install open3d
    pip install tensorboard
    pip install torchmetrics
    pip install hdbscan
    pip install distinctipy
    pip install optuna==3.6.1
    pip install optuna-integration
```

Good luck :)

### Docker
Alternatively, you could simply use docker. Build it first via `make build`, then you can train via doing `make train` and test with `make test CHECKPOINT=<file>`.

