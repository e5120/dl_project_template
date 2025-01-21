from pathlib import Path

import pandas as pd
import polars as pl


def load_raw_data(cfg):
    # interaction
    inter_df = pd.read_csv(
        Path(cfg.data_dir, "ratings.dat"),
        sep="::",
        header=None,
        engine="python",
        nrows=100 if cfg.debug else None,
    )
    inter_df.columns = ["user_id", "item_id", "rating", "timestamp"]
    inter_df = pl.from_pandas(inter_df)
    # user
    user_df = pd.read_csv(
        Path(cfg.data_dir, "users.dat"),
        sep="::",
        header=None,
        engine="python",
        nrows=100 if cfg.debug else None,
    )
    user_df.columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    user_df = pl.from_pandas(user_df)
    # item
    item_df = pd.read_csv(
        Path(cfg.data_dir, "movies.dat"),
        sep="::",
        header=None,
        engine="python",
        encoding="latin1",
        nrows=100 if cfg.debug else None,
    )
    item_df.columns = ["item_id", "title", "genres"]
    item_df = pl.from_pandas(item_df)
    item_df = item_df.with_columns(
        pl.col("title").str.extract(r"(.*)\s\((\d{4})\)", 1).alias("title"),
        pl.col("title").str.extract(r"(.*)\s\((\d{4})\)", 2).str.to_integer().alias("release_year"),
        pl.col("genres").str.split("|"),
    )

    return {
        "inter": inter_df,
        "user": user_df,
        "item": item_df,
    }
