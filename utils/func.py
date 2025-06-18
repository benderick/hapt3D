import torch
import numpy as np
import MinkowskiEngine as ME
from pytorch_lightning.callbacks import EarlyStopping

def TensorField(x, voxel_resolution=0.1):
    """
    Build a tensor field from coordinates and features in the
    tfield batch
    The coordinates are quantized using the provided resolution

    """
    feat_tfield = ME.TensorField(
        features=torch.Tensor(np.concatenate(x["feats"], 0)).float(),
        coordinates=ME.utils.batched_coordinates(
            [np.asarray(i) / voxel_resolution for i in x["points"]], dtype=torch.float32),
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        device="cuda",
    )

    return feat_tfield

class EarlyStoppingWithWarmup(EarlyStopping):
    """
    EarlyStopping, except don't watch the first `warmup` epochs.
    """

    def __init__(self, warmup=10, patience=3, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup
        self.patience = warmup + patience + 1
        self.patience_after_warmup = patience

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch > self.warmup + self.patience_after_warmup + 1:
            self.patience = self.patience_after_warmup
        if trainer.current_epoch < self.warmup:
            return
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)


    def save_pcd_with_predictions():
        import open3d as o3d
        import os
        batch_sensor = batch['sensor'][item]
        filename = batch['path'][item].split('/')[-1]
        res_path = '/media/federico/federico_hdd/hapt3d/results'
        os.makedirs(os.path.join(res_path,batch_sensor),exist_ok=True)
        pts = dense_input.coordinates_at(item).cpu()
        colors = dense_input.features_at(item).cpu()
        ins1 = ins1_preds[item].unsqueeze(-1).cpu()
        ins2 = ins2_preds[item].unsqueeze(-1).cpu()
        sem = sem_pred.unsqueeze(-1).cpu()
        pcd = o3d.t.geometry.PointCloud()
        pcd.point['positions'] = o3d.core.Tensor(pts.numpy())
        pcd.point['colors'] = o3d.core.Tensor(colors.numpy())
        pcd.point['semantic'] = o3d.core.Tensor(sem.numpy().astype(np.int32))
        pcd.point['instance'] = o3d.core.Tensor(ins1.numpy().astype(np.int32))
        pcd.point['instance_h'] = o3d.core.Tensor(ins2.numpy().astype(np.int32))
        o3d.t.io.write_point_cloud(os.path.join(os.path.join(res_path,batch_sensor),filename), pcd)