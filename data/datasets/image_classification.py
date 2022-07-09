from glob import glob
from torch.utils.data import dataset


class ImageClassificationDataset(dataset):
    def __init__(self, root, transforms):
        super(ImageClassificationDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass
