from typing import Union

import numpy as np
import tensorflow as tf


def conv_transpose(w: np.ndarray) -> np.ndarray:
    """Transpose the weights of a PT conv layer so that it's comaptible with TF."""
    return w.transpose(2, 3, 1, 0)


def modify_tf_block(
    tf_component: Union[tf.keras.layers.Layer, tf.Variable, tf.Tensor],
    pt_weight: np.ndarray,
    pt_bias: np.ndarray = None,
    is_attn: bool = False,
) -> Union[tf.keras.layers.Layer, tf.Variable, tf.Tensor]:
    """General utility for modifying PT parameters for TF compatibility.
    Applicable for Conv2D, Dense, tf.Variable, and LayerNormalization.
    """
    pt_weight = (
        conv_transpose(pt_weight)
        if isinstance(tf_component, tf.keras.layers.Conv2D)
        else pt_weight
    )
    pt_weight = (
        pt_weight.transpose()
        if isinstance(tf_component, tf.keras.layers.Dense) and not is_attn
        else pt_weight
    )

    if isinstance(
        tf_component, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)
    ):
        tf_component.kernel.assign(tf.Variable(pt_weight))
        if pt_bias is not None:
            tf_component.bias.assign(tf.Variable(pt_bias))
    elif isinstance(tf_component, tf.keras.layers.LayerNormalization):
        tf_component.gamma.assign(tf.Variable(pt_weight))
        tf_component.beta.assign(tf.Variable(pt_bias))
    elif isinstance(tf_component, (tf.Variable)):
        # For regular variables (tf.Variable).
        tf_component.assign(tf.Variable(pt_weight))
    else:
        return tf.convert_to_tensor(pt_weight)

    return tf_component
