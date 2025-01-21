from src.dataset import BaseDataset


class RatingDataset(BaseDataset):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)

    def __getitem__(self, idx):
        return {
            "user_id": self.df[idx, "user_id"],
            "item_id": self.df[idx, "item_id"],
            "label": self.df[idx, "label"],
        }
