from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from ml_collections import ConfigDict


def conv_transpose(w: np.ndarray) -> np.ndarray:
    """Transpose the weights of a PT conv layer so that it's comaptible with TF."""
    return w.transpose(2, 3, 1, 0)


def modify_attention_block(
    qkv: np.ndarray, config: ConfigDict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modifies the parameters of an attention block so that it's compatible with
    `layers.mha.TFViTSelfAttention`."""
    if qkv.ndim == 2:
        qkv_tf = qkv.T
        q = qkv_tf[:, : config.projection_dim]
        k = qkv_tf[:, config.projection_dim : 2 * config.projection_dim]
        v = qkv_tf[:, -config.projection_dim :]
    elif qkv.ndim == 1:
        qkv_tf = deepcopy(qkv)
        q = qkv_tf[: config.projection_dim]
        k = qkv_tf[config.projection_dim : 2 * config.projection_dim]
        v = qkv_tf[-config.projection_dim :]
    else:
        raise ValueError(
            "NumPy arrays with either two or one dimension are allowed."
        )
    return q, k, v


def get_tf_qkv(
    pt_component: str, pt_params: Dict[str, np.ndarray], config: ConfigDict
):
    """Segregates the query, key, and value subspaces from timm model and makes it
    compatible to be loaded into `layers.mha.TFViTSelfAttention`."""
    qkv_weight = pt_params[f"{pt_component}.qkv.weight"]
    qkv_bias = pt_params[f"{pt_component}.qkv.bias"]

    q_w, k_w, v_w = modify_attention_block(qkv_weight, config)
    q_b, k_b, v_b = modify_attention_block(qkv_bias, config)

    return (q_w, k_w, v_w), (q_b, k_b, v_b)


def modify_tf_block(
    tf_component: tf.keras.layers.Layer,
    pt_weight: np.ndarray,
    pt_bias: np.ndarray = None,
    is_attn: bool = False,
) -> tf.keras.layers.Layer:
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
