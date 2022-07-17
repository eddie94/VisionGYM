from builders import build_transforms

from data.datasets.image_classification import ImageClassificationDataset


def build(config, task):
    if "additional_params" in config.keys():
        assert config.additional_params is not None, "received empty additional_params, " \
                                                     "if you're not using any parameters " \
                                                     "remove the additional_params tab in " \
                                                     "config.yaml"
        additional_params_exists = True
    else:
        additional_params_exists = False

    if task == "classification":
        input_transform = build_transforms.build(
            config.transforms.input,
            **dict(config.additional_params) if additional_params_exists else dict()
        )

        return ImageClassificationDataset(
            root=config.root,
            tf=input_transform
        )
