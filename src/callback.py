from lightning.pytorch.callbacks import *

from src.utils import to_pascal_case


def build_callbacks(cfg):
    import importlib
    return [
        getattr(importlib.import_module(__name__), to_pascal_case(name))(**kwargs)
        for name, kwargs in cfg.items()
    ]
