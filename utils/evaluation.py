import json
import os
import torch
from torchmetrics import JaccardIndex
from torchmetrics.detection import PanopticQuality

TASK_LIST = ['pq', 'pq_h', 'both']

STUFF_IDS = [1, 2, 5]
THINGS_IDS = [3, 4] # fruit and trunk
PQ = torch.Tensor(THINGS_IDS + STUFF_IDS).long()

class Metrics():
    r"""
    Computes iou and panoptic qualities and dumps a json file containing everything. 
    For the 'init': specify task (one of 'pq', 'pq_h', or 'both' for standard panoptic segmentation, hierarchical-only, both at the same time), device and path where to save the json file
    For the 'update': all input tensors should be torch.int64 tensors of shape torch.Size([n_points_in_cloud]). Currently, we only support batch size 1
    In the 'update' function, sensor is a string. Read it from the data you input to the network: the item returned by the dataloader has a key 'sensor'.
    In the 'compute' function, phase is a string (must be either 'val' or 'test'), dump is a bool: if True, will dump a json file with the metrics, if False it will skip (suggestion: dump it when evaluating the final model, and set it to False if you're using this at training time to monitor your metrics), default is True

    Expected predictions for 'pq': 0 void (to be ignored), 1 ground, 2 plant, 3 fruit, 4 trunk, 5 pole
    Expected predictions for 'pq_h': 0 void (to be ignored), 1 tree
    Expected predictions for 'both': two above together
    """
    def __init__(self, task='pq', device='cuda:0', path_to_save=None):
        assert task in TASK_LIST, "Parameter task should be one of `{}`, `{}`, or `{}`".format(*TASK_LIST)
        
        self.task = task
        self.path_to_save_ = path_to_save
        self.sensor_list = ['TLS', 'UAV', 'UGV', 'SFM']
        
        self.metrics = {}
        for s in self.sensor_list:
            self.metrics[s] = {
                'iou': JaccardIndex(task="multiclass", num_classes=6, ignore_index=0, average='none').to(device),
                'pq_h': PanopticQuality(things=[1], stuffs=[0], allow_unknown_preds_category=True, return_sq_and_rq=False, return_per_class=False).to(device),
                'pq': PanopticQuality(things=THINGS_IDS, stuffs=STUFF_IDS, allow_unknown_preds_category=True, return_sq_and_rq=False, return_per_class=True).to(device)
            }

    def update(self, sem_pred=None, sem_target=None, ins_pred=None, ins_target=None, sem_h_pred=None, sem_h_target=None, ins_h_pred=None, ins_h_target=None, sensor='TLS'):
        if self.task == 'both' or self.task == 'pq':

            pan_pred = torch.vstack((sem_pred, ins_pred)).T
            pan_target = torch.vstack((sem_target, ins_target)).T
            jaccard = self.metrics[sensor]['iou']
            jaccard(sem_pred, sem_target)

            pq = self.metrics[sensor]['pq']
            if len(pan_pred.shape) != 3: pan_pred = pan_pred.unsqueeze(0)
            if len(pan_target.shape) != 3: pan_target = pan_target.unsqueeze(0)
            pq(pan_pred, pan_target)

        if self.task == 'both' or self.task == 'pq_h':
            pan_h_pred = torch.vstack((sem_h_pred, ins_h_pred)).T
            pan_h_target = torch.vstack((sem_h_target, ins_h_target)).T
            pq_h = self.metrics[sensor]['pq_h']
            if len(pan_h_pred.shape) != 3: pan_h_pred = pan_h_pred.unsqueeze(0)
            if len(pan_h_target.shape) != 3: pan_h_target = pan_h_target.unsqueeze(0)
            pq_h(pan_h_pred, pan_h_target)

    def compute(self, phase, dump=True): 
        out_dict = {s: {'ious': None, 'miou': None, 'pqs': None, 'mpq': None, 'pqh': None} for s in self.sensor_list}
        
        if self.task == 'both' or self.task == 'pq':       
            for sensor in self.sensor_list:
                ious = self.metrics[sensor]['iou'].compute().cpu()
                pqs = self.metrics[sensor]['pq'].compute()[0].cpu()
                
                # permuting indices to fix torchmetrics
                pqs = pqs[PQ-1]
                
                out_dict[sensor]['ious'] = (ious*100).tolist()
                out_dict[sensor]['miou'] = (ious[1:].mean()*100).item()
                out_dict[sensor]['pqs'] = (pqs*100).tolist()
                out_dict[sensor]['mpq'] = (pqs.mean()*100).item()

        if self.task == 'both' or self.task == 'pq_h':
            for sensor in self.sensor_list:
                pqh_m = self.metrics[sensor]['pq_h']
                if pqh_m.update_count > 0:
                    pqh = pqh_m.compute().cpu()
                else:
                    pqh = torch.Tensor([0])
                out_dict[sensor]['pqh'] = (pqh*100).item()
    
        if dump:
            self.dump(out_dict, phase)
        self.reset()
        return out_dict

    @property
    def path_to_save(self):
        return self.path_to_save_

    @path_to_save.setter
    def path_to_save(self, value):
        self.path_to_save_ = value

    def reset(self):

        for sensor in self.sensor_list:
            self.metrics[sensor]['iou'].reset()
            self.metrics[sensor]['pq'].reset()
            self.metrics[sensor]['pq_h'].reset()

    def dump(self, metrics, phase):
        if self.path_to_save_ is None:
            self.path_to_save_ = "./"
        
        os.makedirs(self.path_to_save_, exist_ok=True)
        with open(os.path.join(self.path_to_save_, 'metrics_{}.json'.format(phase)), 'w') as f:
            json.dump(metrics, f)  

        print("Evaluation completed and file dumped.")      

def main():
    pass