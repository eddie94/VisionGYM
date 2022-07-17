from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS

from typing import Any, Callable, Optional


class ImageClassificationDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            tf: Optional[Callable] = None,
            target_tf: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
    ):
        super(ImageClassificationDataset, self).__init__(
            root=root,
            loader=loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=tf,
            target_transform=target_tf,
            is_valid_file=is_valid_file
        )

        self.imgs = self.samples
