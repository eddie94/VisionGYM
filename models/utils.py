import torch.nn as nn

from typing import Any, List, cast


def make_layers(
        backbone: Any,
        cfg: Any,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "A":
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            layers += [backbone(
                in_channels=in_channels,
                out_channels=v,
            )]
            in_channels = v
    return nn.Sequential(*layers)