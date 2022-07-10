import build_transforms

from data.datasets.image_classification import ImageClassificationDataset


def build(config):
    if config.main.task == "classification":
        input_transform = build_transforms.build(config.dataset.transforms.input)
        gt_transform = build_transforms.build(config.dataset.transforms.gt)

        dataset = ImageClassificationDataset(
            root=config.dataset.root,
            tf_input=input_transform,
            tf_gt=gt_transform
        )

    return dataset
