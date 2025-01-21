import torch.nn as nn

from src.model import BaseModel


class MatrixFactorization(BaseModel):
    def __init__(self, n_users, n_items, embedding_dim=128, label_id="label"):
        super().__init__(label_id=label_id)
        self.loss_fn = nn.MSELoss()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        init_range = 1.0 / embedding_dim
        nn.init.uniform_(self.user_embedding.weight.data, -init_range, init_range)
        nn.init.uniform_(self.item_embedding.weight.data, -init_range, init_range)

    def forward(self, batch):
        user_embs = self.user_embedding(batch["user_id"])
        item_embs = self.item_embedding(batch["item_id"])
        logits = (user_embs * item_embs).sum(dim=1)  # 内積を計算
        return {
            "logits": logits,
        }

    def calculate_loss(self, batch):
        ret = self.forward(batch)
        loss = self.loss_fn(ret["logits"], batch[self.label_id].float())
        ret.update({"loss": loss})
        return ret
