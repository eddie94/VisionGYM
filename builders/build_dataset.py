from builders import build_transforms

from data.datasets.image_classification import ImageClassificationDataset


def build(config, task):
    if task == "classification":
        input_transform = build_transforms.build(
            config.transforms.input,
            **dict(config.additional_params)
        )

        return ImageClassificationDataset(
            root=config.root,
            tf_input=input_transform
        )
