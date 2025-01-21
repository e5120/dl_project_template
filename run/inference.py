# python run/inference.py --experimental-rerun=/path/to/config.pickle
from pathlib import Path

import hydra
import torch
import polars as pl
import lightning as L

import src.model
from src import DataModule, ModelModule
from src.metrics import build_metrics
from src.utils import setup


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg):
    cfg = setup(cfg)
    cfg.stage = "test"
    output_dir = Path(cfg.env.root_dir) if cfg.env.name == "kaggle" else Path(cfg.output_dir)
    model_paths = sorted(output_dir.glob("*.ckpt"))
    datamodule = DataModule(cfg)
    test_loader = datamodule.test_dataloader()
    test_df = test_loader.dataset.df
    is_eval = cfg.LABEL_ID in next(iter(test_loader))
    trainer = L.Trainer(**cfg.trainer)
    results = []
    for fold, model_path in zip(cfg.folds, model_paths, strict=True):
        print(model_path)
        model = getattr(src.model, cfg.model.name)(label_id=cfg.LABEL_ID, **cfg.model.kwargs)
        metrics = build_metrics(cfg.metrics)
        modelmodule = ModelModule.load_from_checkpoint(
            checkpoint_path=model_path,
            model=model,
            metrics=metrics,
            fold=fold,
            cfg=cfg,
        )
        if is_eval:
            results += trainer.test(modelmodule, dataloaders=test_loader)
        predictions = trainer.predict(modelmodule, dataloaders=test_loader)
        predictions = torch.cat(predictions).numpy()
        test_df = test_df.with_columns(pl.Series(predictions).alias("prediction"))
        test_df.write_csv(Path(output_dir, f"submission-{model_path.stem}.csv"))
    if is_eval:
        result_df = pl.DataFrame(results)
        result_df.write_csv(Path(output_dir, "evaluation_results.csv"))
        print(result_df)


if __name__=="__main__":
    main()
