# HOPS 数据格式说明

## 数据集返回格式

### Batch 字典结构

HOPS数据集的DataLoader返回的batch字典包含以下键值对：

```python
batch = {
    # 点云坐标 (主键)
    'positions'   : Tensor, shape=(N, 3), dtype=Float32, device=CPU:0
    
    # RGB颜色值
    'colors'      : Tensor, shape=(N, 3), dtype=Float32, device=CPU:0
    
    # 语义标签 (6类)
    'semantic'    : Tensor, shape=(N, 1), dtype=Float64, device=CPU:0
    
    # 层次化语义标签 (2类: 背景/前景)
    'semantic_h'  : Tensor, shape=(N, 1), dtype=Float64, device=CPU:0
    
    # 实例标签 (fruit + trunk)
    'instance'    : Tensor, shape=(N, 1), dtype=Float64, device=CPU:0
    
    # 树木实例标签 (层次化)
    'instance_h'  : Tensor, shape=(N, 1), dtype=Float64, device=CPU:0
    
    # 其他元数据
    'sensor'      : List[str] - 传感器类型 ['TLS', 'UAV', 'UGV', 'SfM']
    'filename'    : List[str] - 文件名
}
```

### 语义类别定义

```python
SEMANTIC_CLASSES = {
    0: 'void',        # 无效点
    1: 'obstacle',    # 障碍物
    2: 'ground',      # 地面
    3: 'fruit',       # 果实 (things)
    4: 'trunk',       # 树干 (things)
    5: 'foliage'      # 树叶
}

STUFF_IDS = [0, 1, 2, 5]    # stuff类别 (无实例)
THINGS_IDS = [3, 4]          # things类别 (有实例)
```

### 层次化标签

```python
HIERARCHICAL_CLASSES = {
    0: 'background',   # 背景 (void + obstacle + ground + foliage)
    1: 'tree'          # 树木 (fruit + trunk)
}
```

## 模型输出格式

### Forward 函数返回值

```python
output_sem, offsets1, offsets2 = model(dense_input)
```

其中：

1. **output_sem** : `TensorField`
   - 语义分割logits
   - 通过 `output_sem.features_at(batch_id)` 获取 (N, 6) Tensor
   - 每个点有6个类别的置信度分数

2. **offsets1** : `TensorField`
   - 实例偏移向量 (用于fruit和trunk)
   - 通过 `offsets1.features_at(batch_id)` 获取 (N, D) Tensor
   - D通常为3 (xyz偏移) 或更高维度 (embeddings_only=True时)

3. **offsets2** : `TensorField`
   - 树木偏移向量 (用于层次化聚类)
   - 通过 `offsets2.features_at(batch_id)` 获取 (N, D) Tensor
   - 用于将fruit和trunk聚类为完整的tree实例

### 从TensorField获取特征

```python
# 语义预测
sem_logits = output_sem.features_at(batch_id)  # (N, 6)
sem_pred = torch.argmax(sem_logits, dim=1)      # (N,)

# 偏移向量
offsets = offsets1.features_at(batch_id)        # (N, D)
offset_xyz = offsets[:, :3]                     # 取前3维作为xyz偏移

# 坐标 (需要乘以voxel_resolution恢复原始尺度)
coords = dense_input.coordinates_at(batch_id) * voxel_resolution
```

### 后处理：实例聚类

```python
from hdbscan import HDBSCAN

# 计算聚类中心
centers = coords + offset_xyz  # (N, 3)

# HDBSCAN聚类
clusterer = HDBSCAN(
    min_cluster_size=50,      # fruit最小点数
    min_samples=10
)
instance_pred = clusterer.fit_predict(centers)  # (N,) - 实例ID
```

## 数据类型注意事项

### 1. 坐标键名
- ✅ 新版本: `'positions'`
- ⚠️ 旧版本: `'points'`
- 建议: 使用 `'positions' if 'positions' in batch else 'points'` 兼容

### 2. 标签数据类型
- **Ground Truth标签**: `Float64` (需要转换为Int32用于导出)
- **预测标签**: `Int64` (torch.argmax的默认输出)
- **PLY导出**: 需要转换为 `Int32` 或 `Uint32`

```python
# 转换示例
sem_gt = batch['semantic'].squeeze().numpy().astype(np.int32)
ins_gt = batch['instance'].squeeze().numpy().astype(np.int32)
```

### 3. TensorField数据传入

```python
# ✅ 正确: 传入Tensor列表
tensorfield = {
    "points": [points_tensor],
    "feats": [colors_tensor]
}

# ✗ 错误: 传入numpy数组
tensorfield = {
    "points": [points_numpy],
    "feats": [colors_numpy]
}
```

## 常见错误和解决方案

### KeyError: 'coordinates'

**原因**: 可视化脚本使用了错误的键名

**解决**: 使用 `'positions'` 而不是 `'coordinates'` 或 `'points'`

### TypeError: TensorField() got unexpected keyword

**原因**: TensorField期望字典格式 `{"points": [...], "feats": [...]}`

**解决**: 不要传入列表格式 `[points, feats]`

### AttributeError: 'TensorField' object has no attribute 'F'

**原因**: TensorField不再有 `.F` 属性

**解决**: 使用 `.features_at(batch_id)` 方法获取特征

```python
# ✗ 错误
sem_logits = output_sem.F

# ✅ 正确
sem_logits = output_sem.features_at(0)
```

### 数据类型不匹配

**原因**: Ground Truth是Float64，导出PLY需要Int32

**解决**: 显式转换
```python
labels = labels.astype(np.int32)
```

## 完整推理示例

```python
import torch
from utils.func import TensorField
from hdbscan import HDBSCAN

# 1. 准备数据
coords = batch['positions'][0].cuda()  # (N, 3) Float32
colors = batch['colors'][0].cuda()     # (N, 3) Float32

# 2. 创建TensorField
tensorfield = {
    "points": [coords],
    "feats": [colors]
}
dense_input = TensorField(tensorfield, voxel_resolution=0.05)

# 3. 模型推理
with torch.no_grad():
    output_sem, offsets1, offsets2 = model(dense_input)

# 4. 提取预测
sem_logits = output_sem.features_at(0)          # (N, 6)
sem_pred = torch.argmax(sem_logits, dim=1)      # (N,)
offsets = offsets1.features_at(0)[:, :3]        # (N, 3)

# 5. 实例聚类
centers = (dense_input.coordinates_at(0) * 0.05 + offsets).cpu().numpy()
clusterer = HDBSCAN(min_cluster_size=50)
ins_pred = clusterer.fit_predict(centers)

# 6. 转换为numpy并导出
sem_pred_np = sem_pred.cpu().numpy().astype(np.int32)
ins_pred_np = ins_pred.astype(np.int32)
```

## 参考文件

- `models/hapt3d_ours.py` - 模型定义和forward逻辑
- `datasets/dataset.py` - 数据集加载和batch构造
- `export_ply.py` - PLY导出完整示例
- `test_data_keys.py` - 数据格式测试脚本
