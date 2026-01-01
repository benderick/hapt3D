# -*- coding: utf-8 -*-
"""
HAPT3D with HFE, CDAG, HCL - 集成三大创新模块的LightningModule

本文方法:
1. HFE (层次特征增强): 在编码器输出后生成三种专门化特征
2. CDAG (通道双重注意力门控): 在跳跃连接处自适应选择特征
3. HCL (层次一致性损失): 约束实例偏移与树木偏移的层次关系
"""

import torch
import os
import copy
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import JaccardIndex
from torchmetrics.detection import PanopticQuality
from hdbscan import HDBSCAN as hdbscan_cpu
import warnings

from utils.func import TensorField
from utils.lovasz import IoULovaszLoss
from utils.viz import labels2colors
from utils.evaluation import Metrics

STUFF_IDS = [1, 2, 5]
THINGS_IDS = [3, 4]  # fruit and trunk


class HAPT3D(LightningModule):
    """
    HAPT3D: 层次感知的3D果园全景分割网络
    
    支持的模块配置:
    - HFE: network.hfe.enabled
    - CDAG: network.cdag.enabled  
    - HCL: loss.hcl.enabled
    """
    
    def __init__(self, cfg, viz=False):
        super().__init__()
        self.save_hyperparameters(cfg, logger=False)
        self.cfg = cfg
        self.n_classes = cfg['tasks']['semantic_segmentation']['n_classes']
        self.epochs = cfg['train']['max_epoch']
        self.lr = cfg['train']['lr']
        self.voxel_resolution = cfg['train']['voxel_resolution']
        self.ignore_index = cfg['tasks']['semantic_segmentation']['ignore_idx']
        self.embeddings_only = cfg['network']['embeddings_only']
        self.model = self.init_model(cfg)
        self.w_ins1_loss = 1
        self.w_ins2_loss = 1
        self.viz = viz
        self.sensor = 'UAV'

        # ==================== 解析HCL配置 ====================
        self._init_hcl(cfg)

        # ==================== 语义损失 ====================
        sem_count = torch.Tensor([11851+243989, 39009025, 36432676, 1362525, 1017628, 765721])
        sem_weights = torch.Tensor(sum(sem_count)/sem_count)
        sem_weights[0] = 1

        self.sem_loss = torch.nn.CrossEntropyLoss(weight=sem_weights, ignore_index=self.ignore_index)
        self.ins1_loss = IoULovaszLoss(variance=0.01, embeddings_only=self.embeddings_only, fg_classes=THINGS_IDS)
        self.ins2_loss = IoULovaszLoss(variance=0.5, embeddings_only=self.embeddings_only, fg_classes=[1])

        # ==================== 损失累积器 ====================
        self.accumulated_sem_loss = 0.
        self.accumulated_ins1_loss = 0.
        self.accumulated_ins2_loss = 0.
        self.accumulated_hcl_loss = 0.  # 新增: HCL损失累积
        self.iterations = 0

        self.accumulated_sem_loss_val = 0.
        self.accumulated_ins1_loss_val = 0.
        self.accumulated_ins2_loss_val = 0.
        self.accumulated_hcl_loss_val = 0.  # 新增: 验证HCL损失累积
        self.iterations_val = 0

        self.min_n_points_fruit = cfg['val']['min_n_points_fruit']
        self.min_n_points_trunk = cfg['val']['min_n_points_trunk']
        self.min_n_points_tree = cfg['val']['min_n_points_tree']
        self.pq_from_epoch = cfg['val']['pq_from_epoch']
    
        self.jaccard = JaccardIndex(task="multiclass", num_classes=self.n_classes, ignore_index=self.ignore_index, average='none')
        self.pq = PanopticQuality(things=THINGS_IDS, stuffs=STUFF_IDS, allow_unknown_preds_category=True, return_sq_and_rq=False, return_per_class=True)
        self.pq_h = PanopticQuality(things=[1], stuffs=[0], allow_unknown_preds_category=True, return_sq_and_rq=False, return_per_class=False)
        self.metric_eval = Metrics(task='both')

    def _init_hcl(self, cfg):
        """初始化HCL损失模块"""
        # 解析HCL配置
        loss_cfg = cfg.get('loss', {})
        hcl_cfg = loss_cfg.get('hcl', {}) if loss_cfg else {}
        
        self.use_hcl = hcl_cfg.get('enabled', False) if hcl_cfg else False
        self.hcl_weight = hcl_cfg.get('weight', 0.1) if hcl_cfg else 0.1
        
        if self.use_hcl:
            from utils.hcl_loss import HierarchicalConsistencyLoss
            # 初始化HCL损失模块
            self.hcl_loss = HierarchicalConsistencyLoss()
            print(f"[INFO] HCL损失已启用: weight={self.hcl_weight}")
        else:
            self.hcl_loss = None
            print("[INFO] HCL损失已禁用")

    def forward(self, dense_input):
        sparse_input = dense_input.sparse()
        sparse_output_sem, sparse_output_ins1, sparse_output_ins2 = self.model(sparse_input)
        output_sem = sparse_output_sem.slice(dense_input)
        output_ins1 = sparse_output_ins1.slice(dense_input)
        output_ins2 = sparse_output_ins2.slice(dense_input)
        return output_sem, output_ins1, output_ins2

    def getLoss(self, points, logits, sem_target, offsets1, ins1_labels, offsets2, ins2_labels, foreground=None, step='train'):       
        """
        计算总损失
        
        参数:
            points: TensorField，包含点云坐标
            logits: 语义分割logits
            sem_target: 语义标签
            offsets1: 标准实例偏移 (ins1 decoder输出)
            ins1_labels: 标准实例标签
            offsets2: 树木实例偏移 (ins2 decoder输出)
            ins2_labels: 树木实例标签
            foreground: 前景掩码
            step: 'train' 或 'val'
        """
        sem_labels = torch.Tensor(np.concatenate(sem_target, 0)).long().cuda()
        
        # 1. 语义分割损失
        sem_loss = self.sem_loss(logits, sem_labels.squeeze())
        
        # 2. 标准实例分割损失
        ins1_loss = self.ins1_loss(points, ins1_labels, sem_target, offsets1, self.voxel_resolution)
        
        # 3. 树木实例分割损失
        ins2_loss = self.ins2_loss(points, ins2_labels, foreground, offsets2, self.voxel_resolution)
        
        # 4. HCL层次一致性损失（可选）
        hcl_loss = torch.tensor(0.0, device=logits.device)
        if self.use_hcl and self.hcl_loss is not None:
            hcl_loss = self._compute_hcl_loss(points, offsets1, offsets2, ins2_labels, foreground)
        
        # 总损失
        total_loss = sem_loss + self.w_ins1_loss * ins1_loss + self.w_ins2_loss * ins2_loss + self.hcl_weight * hcl_loss
        
        # 累积损失统计
        if step == 'train':
            self.accumulated_sem_loss += sem_loss.detach()
            self.accumulated_ins1_loss += ins1_loss.detach()
            self.accumulated_ins2_loss += ins2_loss.detach()
            if self.use_hcl:
                self.accumulated_hcl_loss += hcl_loss.detach()
            self.iterations += 1
        elif step == 'val':
            self.accumulated_sem_loss_val += sem_loss.detach()
            self.accumulated_ins1_loss_val += ins1_loss.detach()
            self.accumulated_ins2_loss_val += ins2_loss.detach()
            if self.use_hcl:
                self.accumulated_hcl_loss_val += hcl_loss.detach()
            self.iterations_val += 1
            
        return total_loss

    def _compute_hcl_loss(self, points, offsets1, offsets2, ins2_labels, foreground):
        """
        计算HCL层次一致性损失
        
        层次约束: 属于同一棵树的点，其实例偏移应该指向树中心附近
        """
        batch_size = len(foreground)
        total_hcl_loss = torch.tensor(0.0, device=offsets1.F.device)
        valid_batches = 0
        
        for b in range(batch_size):
            # 获取当前batch的数据
            coords_b = points.coordinates_at(b) * self.voxel_resolution  # 恢复原始坐标
            offset_inst_b = offsets1.features_at(b)  # 标准实例偏移
            offset_tree_b = offsets2.features_at(b)  # 树木实例偏移
            tree_labels_b = ins2_labels[b].squeeze()  # 树木实例标签
            foreground_b = foreground[b].squeeze()   # 前景掩码 (0=背景, 1=前景)
            
            # 有效掩码: 前景且有有效树木标签的点
            valid_mask = (foreground_b == 1) & (tree_labels_b > 0)
            
            if valid_mask.sum() < 10:  # 跳过有效点太少的batch
                continue
            
            # 调用HCL损失
            hcl_loss_b = self.hcl_loss(
                coords=coords_b,
                offset_inst=offset_inst_b,
                offset_tree=offset_tree_b,
                tree_labels=tree_labels_b,
                valid_mask=valid_mask
            )
            
            total_hcl_loss += hcl_loss_b
            valid_batches += 1
        
        if valid_batches > 0:
            return total_hcl_loss / valid_batches
        return total_hcl_loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, step="train")

    def training_epoch_end(self, training_step_outputs):
        self.accumulated_sem_loss /= self.iterations
        self.accumulated_ins1_loss /= self.iterations
        self.accumulated_ins2_loss /= self.iterations
        
        self.logger.experiment.add_scalar("Loss/sem_loss", self.accumulated_sem_loss.cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Loss/ins1_loss", self.accumulated_ins1_loss.cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Loss/ins2_loss", self.accumulated_ins2_loss.cpu().numpy(), self.trainer.current_epoch)
        
        # 记录HCL损失
        if self.use_hcl:
            self.accumulated_hcl_loss /= self.iterations
            self.logger.experiment.add_scalar("Loss/hcl_loss", self.accumulated_hcl_loss.cpu().numpy(), self.trainer.current_epoch)
            self.accumulated_hcl_loss *= 0
        
        self.accumulated_sem_loss *= 0
        self.accumulated_ins1_loss *= 0
        self.accumulated_ins2_loss *= 0
        self.iterations = 0

    def validation_step(self, batch, batch_idx):
        self.step(batch, step="val")

    def validation_epoch_end(self, validation_step_outputs):
        self.accumulated_sem_loss_val /= self.iterations_val
        self.accumulated_ins1_loss_val /= self.iterations_val
        self.accumulated_ins2_loss_val /= self.iterations_val
        
        self.logger.experiment.add_scalar("Loss/sem_loss_val", self.accumulated_sem_loss_val.cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Loss/ins1_loss_val", self.accumulated_ins1_loss_val.cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Loss/ins2_loss_val", self.accumulated_ins2_loss_val.cpu().numpy(), self.trainer.current_epoch)
        
        # 记录验证HCL损失
        if self.use_hcl:
            self.accumulated_hcl_loss_val /= self.iterations_val
            self.logger.experiment.add_scalar("Loss/hcl_loss_val", self.accumulated_hcl_loss_val.cpu().numpy(), self.trainer.current_epoch)
            self.accumulated_hcl_loss_val *= 0
        
        ious = self.jaccard.compute()
        miou = ious[1:].mean()
        self.logger.experiment.add_scalar("Metrics_ious/iou1_ground", ious[1].cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Metrics_ious/iou2_plant", ious[2].cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Metrics_ious/iou3_fruit", ious[3].cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Metrics_ious/iou4_trunk", ious[4].cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Metrics_ious/iou5_pole", ious[5].cpu().numpy(), self.trainer.current_epoch)
        self.logger.experiment.add_scalar("Metrics_ious/miou", miou.cpu().numpy().item(), self.trainer.current_epoch)
        self.jaccard.reset()
        
        self.log("Metrics_ious/miou", miou.cpu().numpy().item(), logger=False)
        self.log("Loss/ins1_loss_val", self.accumulated_ins1_loss_val.cpu().numpy().item(), logger=False)
        self.log("Loss/ins2_loss_val", self.accumulated_ins2_loss_val.cpu().numpy().item(), logger=False)
        
        if self.trainer.current_epoch > self.pq_from_epoch:
            pqs = self.pq.compute()[0]
            mpq = pqs.mean()
            self.logger.experiment.add_scalar("Metrics_pqs/mpq", mpq.cpu().numpy().item(), self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq1_ground", pqs[2].cpu().numpy().item(), self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq2_plant", pqs[3].cpu().numpy().item(), self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq3_fruit", pqs[0].cpu().numpy().item(), self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq4_trunk", pqs[1].cpu().numpy().item(), self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq5_pole", pqs[4].cpu().numpy().item(), self.trainer.current_epoch)
            self.pq.reset()

            pqh = self.pq_h.compute()
            self.logger.experiment.add_scalar("Metrics_pqs/pq_h", pqh.cpu().numpy().item(), self.trainer.current_epoch)
            self.pq_h.reset()

            self.log("Metrics_pqs/mpq", mpq.cpu().numpy().item(), logger=False)
            self.log("Metrics_pqs/pq_h", pqh.cpu().numpy().item(), logger=False)

        else:
            self.log("Metrics_pqs/mpq", 0., logger=False)
            self.log("Metrics_pqs/pq_h", 0., logger=False)

            self.logger.experiment.add_scalar("Metrics_pqs/mpq", 0., self.trainer.current_epoch) 
            self.logger.experiment.add_scalar("Metrics_pqs/pq_h", 0., self.trainer.current_epoch) 
            self.logger.experiment.add_scalar("Metrics_pqs/pq1_ground", 0., self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq2_plant", 0., self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq3_fruit", 0., self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq4_trunk", 0., self.trainer.current_epoch)
            self.logger.experiment.add_scalar("Metrics_pqs/pq5_pole", 0., self.trainer.current_epoch)
            
        self.accumulated_sem_loss_val *= 0
        self.accumulated_ins1_loss_val *= 0
        self.accumulated_ins2_loss_val *= 0
        self.iterations_val = 0 

    def test_step(self, batch, batch_idx):
        self.step(batch, step="test", sensor=self.sensor)

    def test_epoch_end(self, outputs):
        if self.metric_eval.path_to_save is None:
            self.metric_eval.path_to_save = self.logger.log_dir
        out_dict = self.metric_eval.compute(phase='test', dump=self.cfg['test']['dump_metrics'])

        pqs = out_dict[self.sensor]['pqs']
        pqh = out_dict[self.sensor]['pqh']
        self.log("Metrics_pqs/pq3_fruit", pqs[2], logger=True)
        self.log("Metrics_pqs/pq4_trunk", pqs[3], logger=True)
        self.log("Metrics_pqs/pq_h", pqh, logger=True)
    
    def step(self, batch, step, sensor='TLS'):
        tensorfield = {"points": batch["points"], "feats": batch["colors"]}
        sem_labels = torch.Tensor(np.concatenate(batch["semantic"], 0)).long().cuda()
        sem_labels_list = [torch.Tensor(item).long().cuda() for item in batch["semantic"]]
        ins1_labels = [torch.Tensor(item).long().cuda() for item in batch["instance"]]
        ins2_labels = [torch.Tensor(item).long().cuda() for item in batch["instance_h"]]
        foreground = [torch.Tensor(item).long().cuda() for item in batch["semantic_h"]]
        dense_input = TensorField(tensorfield,voxel_resolution=self.voxel_resolution)
        output_sem, offsets1, offsets2 = self.forward(dense_input)
        logits = output_sem.F

        loss = self.getLoss(dense_input, logits, batch["semantic"], offsets1, ins1_labels, offsets2, ins2_labels, foreground, step=step)
        if step == "val":
            preds = torch.argmax(logits, dim=1)
            self.jaccard(preds, sem_labels.squeeze())
            
            if self.trainer.current_epoch > self.pq_from_epoch:
                ins1_preds = self.post_processing(batch, output_sem, dense_input, offsets1)
                ins2_preds = self.post_processing(batch, output_sem, dense_input, offsets2, hierarchy=True)

                trunk_ins_mask = copy.deepcopy(ins2_preds)
                for b in range(len(ins1_preds)):
                    sem_pred_batch = torch.argmax(output_sem.features_at(b), dim=1)
                    trunk_ins_mask[b][sem_pred_batch!=4] = 0
                    max_ins = ins1_preds[b].max()
                    ins2_preds[b] += trunk_ins_mask[b] + max_ins

                self.compute_pq(sem_preds=output_sem, sem_targets=sem_labels_list, ins_preds=ins1_preds, ins_targets=ins1_labels)
                self.compute_pq(sem_preds=output_sem, sem_targets=foreground, ins_preds=ins2_preds, ins_targets=ins2_labels, hierarchy=True)
        
        if step == "test":
            if batch['sensor'][0] == sensor:
                preds = torch.argmax(logits, dim=1)
                ins1_preds = self.post_processing(batch, output_sem, dense_input, offsets1)
                ins2_preds = self.post_processing(batch, output_sem, dense_input, offsets2, hierarchy=True)     

                trunk_ins_mask = copy.deepcopy(ins2_preds)
                for b in range(len(ins1_preds)):
                    sem_pred_batch = torch.argmax(output_sem.features_at(b), dim=1)
                    trunk_ins_mask[b][sem_pred_batch!=4] = 0
                    max_ins = ins1_preds[b].max()
                    ins2_preds[b] += trunk_ins_mask[b] + max_ins

                for item in range(len(sem_labels_list)):
                    sem_pred = torch.argmax(output_sem.features_at(item), dim=1)
                    ins = torch.logical_and((ins1_preds[item] == 0).long(), (ins2_preds[item] == 0).long())
                    sem = torch.logical_and((ins == True).long(), (sem_pred==3).long())
                    sem_pred[sem] = 2
                    sem_pred_h = torch.zeros_like(sem_pred)
                    sem_pred_h[torch.logical_or(sem_pred == THINGS_IDS[0], sem_pred == THINGS_IDS[1])] = 1
                

                    self.metric_eval.update(sem_pred=sem_pred, sem_target=sem_labels_list[item].squeeze(), 
                                            ins_pred=ins1_preds[0].squeeze(), ins_target=ins1_labels[item].squeeze(),
                                            sem_h_pred=sem_pred_h.squeeze(), sem_h_target=foreground[item].squeeze(),
                                            ins_h_pred=ins2_preds[item].squeeze(), ins_h_target=ins2_labels[item].squeeze(),
                                            sensor=batch['sensor'][item])

                    if self.viz:
                        pts = dense_input.coordinates_at(item).cpu()
                        labels2colors(pts, sem_pred.cpu(), 'semantic')
                        labels2colors(pts.cpu(), ins1_preds[item].cpu(), 'instance')
                        labels2colors(pts.cpu(), ins2_preds[item].cpu(), 'instance')
            else:
                return loss
        torch.cuda.empty_cache()
        return loss

    def post_processing(self, batch, output_sem, dense_input, offsets, hierarchy=False):
        batch_size = len(batch["points"])
        ins_preds = []
        
        for batch_id in range(batch_size):
            points_batch = dense_input.coordinates_at(batch_id) * self.voxel_resolution
            sem_pred_batch = torch.argmax(output_sem.features_at(batch_id), dim=1)
            things_ids = [1] if hierarchy else THINGS_IDS 
            if hierarchy:
                sem_pred_batch_h = torch.zeros_like(sem_pred_batch)
                sem_pred_batch_h[torch.logical_or(sem_pred_batch == 3, sem_pred_batch == 4)] = 1
                sem_pred_batch = sem_pred_batch_h
            offsets_batch = offsets.features_at(batch_id)
            ins_pred_batch = torch.zeros_like(sem_pred_batch)
            for things_id in things_ids:
                if hierarchy:
                    if things_id == 1: min_n_points = self.min_n_points_tree
                else:
                    if things_id == 3: min_n_points = self.min_n_points_fruit
                category_filter = (sem_pred_batch == things_id)

                if self.embeddings_only:
                    embs_batch = offsets_batch[category_filter]
                else:
                    embs_batch = points_batch[category_filter] + offsets_batch[category_filter]
                if len(embs_batch) <= min_n_points:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clustering = hdbscan_cpu(
                                min_cluster_size=min_n_points,
                                metric="minkowski",
                                p=2.0,
                            ).fit(
                                embs_batch.cpu().numpy()
                            )
                    clusters = clustering.labels_

                clusters_t = (torch.as_tensor(clusters, device=category_filter.device) + 1).bool().long()
                ins_pred_batch[category_filter] += (torch.as_tensor(clusters, device=category_filter.device) + 1 + ins_pred_batch.max()) * clusters_t
            if hierarchy:
                ins_pred_batch[sem_pred_batch != things_ids[0]] = 0
            else:
                ins_pred_batch[~torch.logical_or(sem_pred_batch == things_ids[0], sem_pred_batch == things_ids[1])] = 0

            ins_preds.append(ins_pred_batch)
        return ins_preds

    def compute_pq(self, sem_preds, sem_targets, ins_preds, ins_targets, hierarchy=False):
        batch_size = len(ins_preds)
        for batch_id in range(batch_size):
            sem_pred = torch.argmax(sem_preds.features_at(batch_id), dim=1)
            if hierarchy:
                sem_pred_h = torch.zeros_like(sem_pred)
                sem_pred_h[torch.logical_or(sem_pred == 3, sem_pred == 4)] = 1
                sem_pred = sem_pred_h
            sem_target = sem_targets[batch_id].squeeze()
            ins_pred = ins_preds[batch_id]
            ins_target = ins_targets[batch_id].squeeze()
            pan_pred = torch.vstack((sem_pred, ins_pred)).T
            pan_target = torch.vstack((sem_target, ins_target)).T
            # 将点云数据重塑为伪图像格式 (B, H, W, 2)，其中 H=N, W=1
            # torchmetrics PanopticQuality 期望4D输入
            pan_pred = pan_pred.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
            pan_target = pan_target.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
            if hierarchy:
                self.pq_h(pan_pred, pan_target)
            else:
                self.pq(pan_pred, pan_target)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        return {"optimizer" : self.optimizer, "lr_scheduler": self.scheduler}   

    def init_model(self, cfg):
        """
        根据配置初始化网络模型
        
        支持的配置:
        - network.skip: 跳跃连接类型 (standard/no_skip/decoder_only/full/ours)
        - network.name: 网络名称 (MinkUNet14A/18A/34A/14B/18B/34B等)
        - network.hfe.enabled: 是否启用HFE模块
        - network.cdag.enabled: 是否启用CDAG模块
        """
        use_tanh = cfg['network']['tanh']
        skip_type = cfg['network']['skip']
        
        # 检查是否使用我们的模块 (HFE/CDAG)
        use_ours = True
        # hfe_cfg = cfg.get('network', {}).get('hfe', {})
        # cdag_cfg = cfg.get('network', {}).get('cdag', {})
        
        # if hfe_cfg and hfe_cfg.get('enabled', False):
        #     use_ours = True
        # if cdag_cfg and cdag_cfg.get('enabled', False):
        #     use_ours = True
        
        # 选择网络模块
        if use_ours or skip_type == 'ours':
            # 使用我们集成HFE/CDAG的网络
            from .minkunet_ours import (
                MinkUNet14A_Ours as MinkUNet14A,
                MinkUNet18A_Ours as MinkUNet18A,
                MinkUNet34A_Ours as MinkUNet34A,
                MinkUNet14B_Ours as MinkUNet14B,
                MinkUNet18B_Ours as MinkUNet18B,
                MinkUNet34B_Ours as MinkUNet34B,
            )
            # 额外定义缺失的变体
            MinkUNet14C = MinkUNet14A
            MinkUNet34C = MinkUNet34A
            MinkUNet14D = MinkUNet14A
            MinkUNet18D = MinkUNet18A
            need_cfg = True  # 需要传递cfg
            print("[INFO] 使用集成HFE/CDAG的MinkUNet (minkunet_ours)")
        elif skip_type == 'standard':
            from .minkunet import MinkUNet14A, MinkUNet18A, MinkUNet34A, MinkUNet14B, MinkUNet18B, MinkUNet34B, MinkUNet14C, MinkUNet34C, MinkUNet14D, MinkUNet18D
            need_cfg = False
        elif skip_type == 'no_skip':
            from .minkunet_no_skip import MinkUNet14A, MinkUNet18A, MinkUNet34A, MinkUNet14B, MinkUNet18B, MinkUNet34B, MinkUNet14C, MinkUNet34C, MinkUNet14D, MinkUNet18D
            need_cfg = False
        elif skip_type == 'decoder_only':
            from .minkunet_decoder_only import MinkUNet14A, MinkUNet18A, MinkUNet34A, MinkUNet14B, MinkUNet18B, MinkUNet34B, MinkUNet14C, MinkUNet34C, MinkUNet14D, MinkUNet18D
            need_cfg = False
        elif skip_type == 'full':
            from .minkunet_full import MinkUNet14A, MinkUNet18A, MinkUNet34A, MinkUNet14B, MinkUNet18B, MinkUNet34B, MinkUNet14C, MinkUNet34C, MinkUNet14D, MinkUNet18D
            need_cfg = False
        else:
            raise NotImplementedError('{} not implemented'.format(skip_type))
        
        # 选择具体网络 (兼容 'name' 和 'backbone' 两种配置方式)
        network_cfg = cfg.get('network', {})
        net_name = network_cfg.get('name') or network_cfg.get('backbone', 'MinkUNet14A')
        net_dict = {
            'MinkUNet14A': MinkUNet14A,
            'MinkUNet18A': MinkUNet18A,
            'MinkUNet34A': MinkUNet34A,
            'MinkUNet14B': MinkUNet14B,
            'MinkUNet18B': MinkUNet18B,
            'MinkUNet34B': MinkUNet34B,
            'MinkUNet14C': MinkUNet14C,
            'MinkUNet34C': MinkUNet34C,
            'MinkUNet14D': MinkUNet14D,
            'MinkUNet18D': MinkUNet18D,
        }
        
        if net_name not in net_dict:
            raise NotImplementedError('{} not implemented'.format(net_name))
        
        net = net_dict[net_name]
        
        # 实例化网络
        if need_cfg:
            model = net(3, self.n_classes, cfg=cfg, instance_decoder=True, use_tanh=use_tanh)
        else:
            model = net(3, self.n_classes, instance_decoder=True, use_tanh=use_tanh)
        
        return model
