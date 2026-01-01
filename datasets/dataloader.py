from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import HAPT3DDataset


class StatDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 兼容两种配置格式: data_path (旧) 或 data.path (新)
        if "data_path" not in self.cfg:
            self.cfg["data_path"] = self.cfg.get("data", {}).get("path", "data/hopt3d")
        self.setup()
        self.loader = [ self.train_dataloader(), self.val_dataloader() ]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = HAPT3DDataset(data_path=self.cfg["data_path"], config=self.cfg, split="train", overfit=self.cfg['train']['overfit'])
            self.data_val = HAPT3DDataset(data_path=self.cfg["data_path"], config=self.cfg, split="train" if self.cfg['train']['overfit'] else "val", overfit=self.cfg['train']['overfit'])
            self.data_test = HAPT3DDataset(data_path=self.cfg["data_path"], config=self.cfg, split="train" if self.cfg['train']['overfit'] else "test", overfit=self.cfg['train']['overfit'])

    def train_dataloader(self):
        loader = DataLoader(self.data_train, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            collate_fn=self.data_train.collate,
                            num_workers = self.cfg['train']['workers'],
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data_val, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            collate_fn=self.data_train.collate,
                            num_workers = self.cfg['train']['workers'],
                            shuffle=False)
        self.len = self.data_train.__len__()
        return loader
            
    def test_dataloader(self):
        loader = DataLoader(self.data_test, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            collate_fn=self.data_train.collate,
                            num_workers = self.cfg['train']['workers'],
                            shuffle=False)
        self.len = self.data_train.__len__()
        return loader
