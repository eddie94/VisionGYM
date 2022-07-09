from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(self):
        super(DataModule, self).__init__()
