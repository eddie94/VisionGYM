import torchvision.transforms as tf

from torchvision.transforms import Compose


def simple_tf():
    return Compose(
        tf.PILToTensor()
    )


def simple_tf_rescale():
    return Compose(
        tf.ToTensor()
    )
