from data.transforms import pre_defined_transforms as tf


def build(tf_type):
    if tf_type == "simple_tf":
        transform = tf.simple_tf()
    elif tf_type == "simple_tf_rescale":
        transform = tf.simple_tf_rescale()

    return transform
