from pathlib import Path

import polars as pl
import lightning as L
from torch.utils.data import DataLoader

import src.dataset


class DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir, cfg.dataset_name)
        self.fold = -1
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        # define Dataset class
        self.dataset_cls = getattr(src.dataset, self.cfg.dataset.name)
        # load dataset
        self.main_df = self.load_data()

    def reset(self, fold):
        self.fold = fold

    def load_data(self):
        df = pl.read_parquet(Path(self.data_dir, f"{self.cfg.stage}.parquet"))
        return df

    def _generate_dataset(self, stage):
        if stage == "train":
            df = self.main_df.filter(pl.col("fold") != self.fold)
        elif stage == "val":
            df = self.main_df.filter((pl.col("fold") == self.fold))
        elif stage == "test":
            df = self.main_df
        else:
            raise NotImplementedError
        dataset = self.dataset_cls(df, **self.cfg.dataset.kwargs)
        return dataset

    def _generate_dataloader(self, dataset, stage):
        if stage == "train":
            shuffle=True
            drop_last=True
        else:
            shuffle=False
            drop_last=False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )

    def train_dataloader(self):
        train_dataset = self._generate_dataset("train")
        train_loader = self._generate_dataloader(train_dataset, "train")
        return train_loader

    def val_dataloader(self):
        val_dataset = self._generate_dataset("val")
        val_loader = self._generate_dataloader(val_dataset, "val")
        return val_loader

    def test_dataloader(self):
        test_dataset = self._generate_dataset("test")
        test_loader = self._generate_dataloader(test_dataset, "test")
        return test_loader

    def predict_dataloader(self):
        return self.test_dataloader()
