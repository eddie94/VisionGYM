from data.datasets.image_classification import ImageClassificationDataset


def build_image_classification_dataset(config):
    dataset = ImageClassificationDataset(
        root=config.root,
        transforms=config.transforms,
    )

    return dataset
