from typing import Any
from collections.abc import Iterable


class Params:
    def __init__(self) -> None:
        self.params = {}
    
    def add_param(
        self,
        name: str,
        value: Any,
    ):
        self.params[name] = value
        
    def __getitem__(self, i):
        param_dict = {}
        
        for key in self.params.keys():
            if isinstance(self.params[key], Iterable):
                param_dict[key] = self.params[key][i]
            else:
                param_dict[key] = self.params[key]
        
        return param_dict


def assert_conv2d_options(config):
    pass
