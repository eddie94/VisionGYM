import torch.nn as nn


class VggBackbone(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(VggBackbone, self).__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )


class VGGClassifier(nn.Sequential):
    def __init__(
            self,
            num_classes: int,
            in_features: int = 512 * 7 * 7,
    ):
        super(VGGClassifier, self).__init__(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
