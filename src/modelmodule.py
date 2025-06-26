import lightning as L

import src.optimizer
import src.scheduler


class ModelModule(L.LightningModule):
    def __init__(self, model, metrics, fold, cfg):
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.index_id = cfg.INDEX_ID
        self.pred_id = cfg.PRED_ID
        self.label_id = cfg.LABEL_ID
        self.fold = fold
        self.cfg = cfg

    def forward(self, batch):
        return self.model.forward(batch)

    def calculate_loss(self, batch):
        return self.model.calculate_loss(batch)

    def training_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch)
        self.log("loss", ret["loss"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        # for param_group in self.trainer.optimizers[0].param_groups:
        #     lr = param_group["lr"]
        # self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return ret["loss"]

    def validation_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch)
        self.log("val_loss", ret["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.index_id is None:
            self.metrics.update(ret[self.pred_id], batch[self.label_id])
        else:
            self.metrics.update(ret[self.pred_id], batch[self.label_id], indexes=batch[self.index_id])

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.metrics.reset()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("fold", self.fold, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ret = self.forward(batch)
        return ret[self.pred_id]

    def configure_optimizers(self):
        optimizer = getattr(src.optimizer, self.cfg.optimizer.name)(
            self.parameters(),
            **self.cfg.optimizer.kwargs,
        )
        scheduler = getattr(src.scheduler, self.cfg.scheduler.name)(
            optimizer,
            **self.cfg.scheduler.kwargs,
        )
        interval = "epoch" if self.cfg.scheduler.name in ["ReduceLROnPlateau"] else "step"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": self.cfg.monitor,
            }
        }
