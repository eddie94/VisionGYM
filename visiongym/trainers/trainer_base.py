from typing import Any
from lightning.pytorch import LightningModule
from visiongym.builders.backbone import BACKBONE_REGISTRY


class TrainerBase(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.model_cfg = config.model
        self.backbone_cfg = self.model_cfg.backbone
        
        self.backbone = BACKBONE_REGISTRY.get(self.backbone_cfg.name)
