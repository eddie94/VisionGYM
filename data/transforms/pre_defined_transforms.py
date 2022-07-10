import torchvision.transforms as tf

from torchvision.transforms import Compose


def simple_tf():
    return Compose([
        tf.PILToTensor()
    ])


def simple_tf_rescale():
    return Compose([
        tf.ToTensor()
    ])


def standard_tf(
        img_size=64,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        p=0.5
):
    return Compose([
        tf.ToTensor(),
        tf.Resize(img_size),
        tf.RandomHorizontalFlip(p=p),
        tf.Normalize(mean=mean, std=std),
    ])
