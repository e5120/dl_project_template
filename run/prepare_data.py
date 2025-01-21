from pathlib import Path

import hydra
import polars as pl
from sklearn.model_selection import train_test_split, KFold

from src.data import load_raw_data
from src.utils import setup


@hydra.main(config_path="conf", config_name="prepare_data", version_base=None)
def main(cfg):
    cfg = setup(cfg)
    data = load_raw_data(cfg)
    inter_df = (
        data["inter"]
        .select(
            pl.col("user_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Int32),
            pl.col("rating").cast(pl.Int32).alias("label"),
        )
    )
    train_idx, test_idx = train_test_split(range(len(inter_df)), test_size=0.1, random_state=cfg.seed)
    df = inter_df[train_idx]
    test_df = inter_df[test_idx]
    kfold = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    df = df.to_pandas()
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kfold.split(df)):
        df.loc[val_idx, "fold"] = fold
    df = pl.from_pandas(df)
    df.write_parquet(Path(cfg.output_dir, "train.parquet"))
    test_df.write_parquet(Path(cfg.output_dir, "test.parquet"))
    for fold in range(cfg.n_folds):
        trn_df = df.filter(pl.col("fold") != fold)
        val_df = df.filter(pl.col("fold") == fold)
        print(f"[fold {fold}] # of train: {len(trn_df):,}, # of val: {len(val_df):,}")
    print(f"# of test: {len(test_df):,}")


if __name__=="__main__":
    main()
