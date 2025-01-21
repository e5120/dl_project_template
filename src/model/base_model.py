import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, label_id="label"):
        super().__init__()
        self.label_id = label_id

    def forward(self, batch):
        raise NotImplementedError

    def calculate_loss(self, batch):
        raise NotImplementedError
