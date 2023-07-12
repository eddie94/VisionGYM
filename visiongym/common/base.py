import torch.nn as nn
import visiongym.common.layers as layers

from abc import ABCMeta, abstractmethod


class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def build_module(self):
        pass
    
    @staticmethod
    def build_conv2d_layer(deformable):
        # deformable conv settings
        if deformable:
            return layers.DeformableConv2d
        else:
            return nn.Conv2d
