from models.classification.vgg import *


def build(config):
    return globals()[config.type]
