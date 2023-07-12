from typing import Any
from collections.abc import Iterable
from omegaconf import OmegaConf


class Params:
    def __init__(self, param_dict: dict = {}) -> None:
        self.params = param_dict
    
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
    

def assert_param_len(*params) -> bool:
    """returns if all the parameters have the same length

    Returns:
        true if all parameters have the same length, false if not
    """
    it = iter(params)
    param_len = len(next(it))
    if not all(len(l) == param_len for l in it):
        return False
    else:
        return True


def assert_conv2d_options(config: OmegaConf) -> Params:
    if "additional_params" in config:
        param_dict = config.additional_params
        assert assert_param_len(param_dict.values()), "parameters don't hane the same length"
    else:
        param_dict = {}
    
    return Params(param_dict)
