import torch.nn as nn


def build(config):
    if "params" in config.keys():
        return getattr(nn, config.type)(**config.params)
    else:
        return getattr(nn, config.type)()
