from data.transforms import pre_defined_transforms as tf


def build(tf_type, **additional_params):
    if tf_type == "simple_tf":
        return tf.simple_tf()
    elif tf_type == "simple_tf_rescale":
        return tf.simple_tf_rescale()
    elif tf_type == "standard_tf":
        return tf.standard_tf(**additional_params)
