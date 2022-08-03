from torch.optim.adam import *


def build(config):
    if "params" in config.keys():
        return globals()[config.type](**config.params)
    else:
        return globals()[config.type]()
