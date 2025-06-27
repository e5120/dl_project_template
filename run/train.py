import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger

import src.model
from src import ModelModule, DataModule
from src.callback import build_callbacks
from src.metrics import build_metrics
from src.utils import setup, get_num_training_steps


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg):
    cfg = setup(cfg)
    datamodule = DataModule(cfg)
    for fold in cfg.folds:
        datamodule.reset(fold)
        # スケジューラのパラメータを設定
        num_train_data = len(datamodule._generate_dataset("train"))
        max_steps = get_num_training_steps(num_train_data, cfg)
        if "num_training_steps" in cfg.scheduler.kwargs:
            cfg.scheduler.kwargs.num_training_steps = max_steps
        elif "T_max" in cfg.scheduler.kwargs:
            cfg.scheduler.kwargs.T_max = max_steps
        elif "total_steps" in cfg.scheduler.kwargs:
            cfg.scheduler.kwargs.total_steps = max_steps
        # モデルの構築＆学習
        model = getattr(src.model, cfg.model.name)(label_id=cfg.LABEL_ID, **cfg.model.kwargs)
        metrics = build_metrics(cfg.metrics)
        modelmodule = ModelModule(model, metrics, fold, cfg)
        callbacks = build_callbacks(cfg.callbacks)
        trainer = L.Trainer(
            callbacks=callbacks,
            logger=WandbLogger(name=cfg.exp_name, project=cfg.project_name) if cfg.project_name and cfg.exp_name != "dummy" else None,
            **cfg.trainer,
        )
        trainer.fit(modelmodule, datamodule)


if __name__=="__main__":
    main()
