from torchmetrics import MetricCollection
from torchmetrics import *
from torchmetrics.retrieval import *


def build_metrics(cfg):
    import importlib
    metrics = {}
    for name, kwargs in cfg.items():
        kwargs = dict(kwargs)
        alias = kwargs.pop("alias") if "alias" in kwargs else name
        alias = alias.lower()
        metrics[alias] = getattr(importlib.import_module(__name__), name)(**kwargs)
    metrics = MetricCollection(metrics)
    return metrics
