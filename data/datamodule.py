from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule

from builders import build_dataset


class DataModule(LightningDataModule):
    def __init__(self, config):
        super(DataModule, self).__init__()

        self.config = config
        self.task = config.main.task

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self.trainset = build_dataset.build(self.config.dataset.train, self.task)
            self.valset = build_dataset.build(self.config.dataset.val, self.task)

        if stage == "test" or stage is None:
            self.testset = build_dataset.build(self.config.dataset.test, self.task)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset,
            batch_size=self.config.dataset.train.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valset,
            batch_size=self.config.dataset.val.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset,
            batch_size=self.config.dataset.test.batch_size
        )
