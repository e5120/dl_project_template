import re

import hydra
import lightning as L


_PARSE_BY_SEP_PATTERN = re.compile(r"[ _-]+")
# _PARSE_PATTERN = re.compile(r"[A-Za-z][^A-Z]+")
_PARSE_PATTERN = re.compile(r"[A-Za-z]+")


def _parse_words(string):
    for block in re.split(_PARSE_BY_SEP_PATTERN, string):
        for m in re.finditer(_PARSE_PATTERN, block):
            yield m.group(0)


def to_pascal_case(string):
    word_iter = _parse_words(string)
    return "".join(word.capitalize() for word in word_iter)


def setup(cfg):
    cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if "trainer" in cfg and cfg.trainer.default_root_dir:
        cfg.trainer.default_root_dir = cfg.output_dir
    # コールバック関数に関する前処理
    if "callbacks" in cfg:
        if "early_stopping" in cfg.callbacks:
            cfg.callbacks.early_stopping.monitor = cfg.monitor
            cfg.callbacks.early_stopping.mode = cfg.mode
        if "model_checkpoint" in cfg.callbacks:
            kwargs = dict(cfg.callbacks.model_checkpoint)
            prefix = kwargs.pop("prefix")
            digit = kwargs.pop("digit")
            performance = "{" + f"{cfg.monitor}:.{digit}f" + "}"
            kwargs["filename"] = f"{prefix}-fold-{{fold:.0f}}-epoch-{{epoch:03d}}-{cfg.monitor}-{performance}"
            kwargs["monitor"] = cfg.monitor
            kwargs["mode"] = cfg.mode
            kwargs["dirpath"] = cfg.output_dir
            cfg.callbacks.model_checkpoint = kwargs
    L.seed_everything(cfg["seed"])
    return cfg


def get_num_training_steps(n_data, cfg):
    num_devices = 1 if isinstance(cfg.trainer.devices, int) else len(cfg.trainer.devices)
    steps_per_epoch = n_data // cfg.batch_size // num_devices // cfg.trainer.accumulate_grad_batches
    num_training_steps = steps_per_epoch * cfg.trainer.max_epochs
    return num_training_steps
