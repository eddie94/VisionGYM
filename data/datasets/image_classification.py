from glob import glob
from torch.utils.data.dataset import Dataset


class ImageClassificationDataset(Dataset):
    def __init__(self, root, tf_input, tf_gt):
        super(ImageClassificationDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass
