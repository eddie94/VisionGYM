import torch.nn as nn

from models.utils import make_layers
from models.backbone import vgg


class VGGBase(nn.Module):
    def __init__(self, config):
        super(VGGBase, self).__init__()

        self.cfgs = {
            "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        if config.type in self.cfgs.keys():
            cfgs = self.cfgs[config.type]
        else:
            cfgs = config.cfgs

        self.convs = make_layers(vgg.VggBackbone, cfgs)
        self.fc = getattr(vgg, config.classifier.type)(**config.classifier.params)

    def forward(self, x):
        raise NotImplementedError()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG11(VGGBase):
    def __init__(self, config):
        super(VGG11, self).__init__(config=config)
        self._initialize_weights()

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)

        return x


class VGG13(VGGBase):
    def __init__(self, config):
        super(VGG13, self).__init__(config=config)
        self._initialize_weights()

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)

        return x


class VGG16(VGGBase):
    def __init__(self, config):
        super(VGG16, self).__init__(config=config)
        self._initialize_weights()

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)

        return x


class VGG19(VGGBase):
    def __init__(self, config):
        super(VGG19, self).__init__(config=config)
        self._initialize_weights()

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)

        return x


class VGGCustom(VGGBase):
    def __init__(self, config):
        super(VGGCustom, self).__init__(config=config)
        self._initialize_weights()

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)

        return x
