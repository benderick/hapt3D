import os
import copy

import numpy as np
import torch
import open3d as o3d

from torch.utils.data import Dataset

from datasets.tf import get_geom_aug, color_jitter, elastic_distortion

'''
SEMANTIC:
    0: void, unlabeled, "to_check" --> ignored in metrics and losses
    1: ground [stuff]
    2: plant  [stuff]
    3: fruit  [thing]
    4: trunk  [thing]
    5: pole   [stuff]

SEMANTIC_H:
    0: void, unlabeled, "to_check" --> ignored in metrics and losses 
    1: plant (ie, trunk+fruits) [thing]
'''

class HAPT3DDataset(Dataset):
    def __init__(self, data_path, config, split='train', overfit=False):
        sensor_folders = os.listdir(os.path.join(data_path, split))
        self.split = split
        self.cfg = config
        self.overfit = overfit
        self.pcd_full_path = []
        
        if self.overfit:
            self.split = 'train'
        
        for sensor in sensor_folders:
            sensor_path = os.path.join(data_path, split, sensor)
            pcd_list = os.listdir(sensor_path)
            self.pcd_full_path += absPath(sensor_path, pcd_list)      

        self.do_augmentation = True if self.split == 'train' and not self.overfit else False  

    def __len__(self):
        if self.overfit:
            return 1
        return len(self.pcd_full_path)

    def __getitem__(self, index):
        item = {}
        pcd_path = self.pcd_full_path[index]
        item['path'] = pcd_path
        item['sensor'] = pcd_path.split('/')[-2]
        pcd = o3d.t.io.read_point_cloud(self.pcd_full_path[index])
        orig_pcd = copy.deepcopy(pcd)
        
        # downsample
        if self.do_augmentation:
            sampling_ratio = np.random.uniform(self.cfg['transform']['downsample']['min_ratio'], self.cfg['transform']['downsample']['max_ratio']) 
            pcd = pcd.random_down_sample(sampling_ratio=sampling_ratio)
        
        pcd = self.translate_pcd(pcd)

        # geometric augmentations
        if self.do_augmentation:
            T = get_geom_aug(self.cfg)
            pcd.transform(T)
            colors = color_jitter(self.cfg, pcd.point["colors"])
            pcd.point["colors"] = colors
            points = pcd.point['positions'].numpy()
            granularity = np.random.uniform(low=0.01, high=0.1)
            try:
                points = elastic_distortion(points.astype('float32'), granularity, granularity*2)
                pcd.point['positions'] = o3d.core.Tensor(points)
            except IndexError:
                pass

        pcd = self.normalize_pcd(pcd)

        item['points'] = self.o3d2torch(pcd.point['positions'], 'float32')
        item['colors'] = self.o3d2torch(pcd.point['colors'], 'float32')
        # if False: #self.split == 'test':
        #     item['semantic'] = torch.zeros((len(item['points']), 1))
        #     item['instance'] = torch.zeros((len(item['points']), 1))
        #     item['semantic_h'] = torch.zeros((len(item['points']), 1))
        #     item['instance_h'] = torch.zeros((len(item['points']), 1))
        # else:
        item['semantic'] = self.o3d2torch(pcd.point['semantic'])
        item['instance'] = self.o3d2torch(pcd.point['instance'])
        item['semantic_h'] = self.o3d2torch(pcd.point['semantic_h'])
        item['instance_h'] = self.o3d2torch(pcd.point['instance_h'])

        # -1 to ignore unlabeled data and have void = 0, classes = 1, ..., K
        # semantic == 6 is "UNKNOWN_TO_CHECK" from segments (shouldn't appear at all)
        item['semantic'] -= 1
        item['semantic'][(item['semantic'] < 0)] = -1
        item['semantic'][(item['semantic'] >= 6)] = -1
        item['semantic_h'] -= 1

        item['semantic'][(item['semantic'] < 0)] = 0
        item['semantic_h'][(item['semantic_h'] < 0)] = 0

        return item

    @staticmethod
    def collate(batch):
        item = {}
        for key in batch[0].keys():
            item[key] = []

        for key in item.keys():
            for data in batch:
                if isinstance(data[key], str):
                    item[key].append(data[key])
                else:
                    item[key].append(data[key].numpy())
        return item

    @staticmethod
    def o3d2torch(o3dvec, dtype='uint8'):
        return torch.Tensor(o3dvec.numpy().astype(dtype))
    
    @staticmethod
    def translate_pcd(pcd):
        # centering
        semantic = pcd.point['semantic'].numpy()[:, 0]
        soil = pcd.point['positions'].numpy()[semantic == 2]
        mean_soil = np.mean(soil, axis=0)

        pcd = pcd.translate(-mean_soil)
        
        return pcd

    @staticmethod
    def normalize_pcd(pcd):
        pts = pcd.point['positions']

        maxb =  pcd.get_max_bound()[-1]
        minb =  pcd.get_min_bound()[-1]
        pts_scaled = (pts - minb) / (maxb - minb)
        pcd.point['positions'] = pts_scaled
        return pcd

def absPath(data_path, file_list):
    return [os.path.join(data_path, f) for f in file_list]
