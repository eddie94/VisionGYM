import torchmetrics


def build(config):
    if "params" in config.keys():
        return getattr(torchmetrics, config.type)(**config.params)
    else:
        return getattr(torchmetrics, config.type)()