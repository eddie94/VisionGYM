import torch.nn as nn

from abc import ABCMeta, abstractmethod


class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self, config) -> None:
        super().__init__()
        
    @abstractmethod
    def forward(self):
        pass