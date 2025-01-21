from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError
