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

    def setup(self, stage=None):
        """
        PyTorch Lightning 数据准备流程：
        - stage='fit': 训练时调用，准备 train 和 val 数据集
        - stage='test': 测试时调用，准备 test 数据集
        - stage=None: 准备所有数据集
        """
        # 训练和验证数据集
        if stage == 'fit' or stage is None:
            self.data_train = HAPT3DDataset(
                data_path=self.cfg["data_path"], 
                config=self.cfg, 
                split="train", 
                overfit=self.cfg['train']['overfit']
            )
            self.data_val = HAPT3DDataset(
                data_path=self.cfg["data_path"], 
                config=self.cfg, 
                split="train" if self.cfg['train']['overfit'] else "val", 
                overfit=self.cfg['train']['overfit']
            )
        
        # 测试数据集
        if stage == 'test' or stage is None:
            self.data_test = HAPT3DDataset(
                data_path=self.cfg["data_path"], 
                config=self.cfg, 
                split="train" if self.cfg['train'].get('overfit', False) else "test", 
                overfit=self.cfg['train'].get('overfit', False)
            )

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
                            collate_fn=self.data_val.collate,
                            num_workers = self.cfg['train']['workers'],
                            shuffle=False)
        return loader
            
    def test_dataloader(self):
        loader = DataLoader(self.data_test, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            collate_fn=self.data_test.collate,
                            num_workers = self.cfg['train']['workers'],
                            shuffle=False)
        return loader
