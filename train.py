import hydra

from data.datamodule import DataModule


@hydra.main(version_base=None, config_path="configs/sample", config_name="image_classification_config")
def main(config) -> None:
    datamodule = DataModule(config=config)


if __name__ == '__main__':
    main()
