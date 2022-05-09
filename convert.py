"""
Some code is copied from here:
https://github.com/sayakpaul/cait-tf/blob/main/convert.py
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import tensorflow as tf
import timm

sys.path.append("..")


from swins import SwinTransformer, model_configs
from swins.blocks import *
from swins.layers import *
from utils import helpers

TF_MODEL_ROOT = "gs://swin-tf"

NUM_CLASSES = {"in1k": 1000, "in21k": 21841}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained Swin weights to TensorFlow."
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="swin_tiny_patch4_window7_224",
        type=str,
        choices=model_configs.MODEL_MAP.keys(),
        help="Name of the Swin model variant.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="in1k",
        choices=["in1k", "in21k"],
        type=str,
    )
    parser.add_argument(
        "-pl",
        "--pre-logits",
        action="store_true",
        help="If we don't need the classification outputs.",
    )
    return parser.parse_args()


def modify_swin_blocks(
    np_state_dict: Dict[str, np.ndarray],
    pt_weights_prefix: str,
    tf_block: List[tf.keras.layers.Layer],
) -> List[tf.keras.layers.Layer]:
    """Main utility to convert params of a swin block."""
    # Patch merging.
    for layer in tf_block:
        if isinstance(layer, PatchMerging):
            patch_merging_idx = f"{pt_weights_prefix}.downsample"

            layer.reduction = helpers.modify_tf_block(
                layer.reduction,
                np_state_dict[f"{patch_merging_idx}.reduction.weight"],
            )
            layer.norm = helpers.modify_tf_block(
                layer.norm,
                np_state_dict[f"{patch_merging_idx}.norm.weight"],
                np_state_dict[f"{patch_merging_idx}.norm.bias"],
            )

    # Swin layers.
    common_prefix = f"{pt_weights_prefix}.blocks"
    block_idx = 0

    for outer_layer in tf_block:

        layernorm_idx = 1
        mlp_layer_idx = 1

        if isinstance(outer_layer, SwinTransformerBlock):
            for inner_layer in outer_layer.layers:

                # Layer norm.
                if isinstance(inner_layer, tf.keras.layers.LayerNormalization):
                    layer_norm_prefix = (
                        f"{common_prefix}.{block_idx}.norm{layernorm_idx}"
                    )
                    inner_layer.gamma.assign(
                        tf.Variable(
                            np_state_dict[f"{layer_norm_prefix}.weight"]
                        )
                    )
                    inner_layer.beta.assign(
                        tf.Variable(np_state_dict[f"{layer_norm_prefix}.bias"])
                    )
                    layernorm_idx += 1

                # Windown attention.
                elif isinstance(inner_layer, WindowAttention):
                    attn_prefix = f"{common_prefix}.{block_idx}.attn"

                    # Relative position.
                    inner_layer.relative_position_bias_table = (
                        helpers.modify_tf_block(
                            inner_layer.relative_position_bias_table,
                            np_state_dict[
                                f"{attn_prefix}.relative_position_bias_table"
                            ],
                        )
                    )
                    inner_layer.relative_position_index = (
                        helpers.modify_tf_block(
                            inner_layer.relative_position_index,
                            np_state_dict[
                                f"{attn_prefix}.relative_position_index"
                            ],
                        )
                    )

                    # QKV.
                    inner_layer.qkv = helpers.modify_tf_block(
                        inner_layer.qkv,
                        np_state_dict[f"{attn_prefix}.qkv.weight"],
                        np_state_dict[f"{attn_prefix}.qkv.bias"],
                    )

                    # Projection.
                    inner_layer.proj = helpers.modify_tf_block(
                        inner_layer.proj,
                        np_state_dict[f"{attn_prefix}.proj.weight"],
                        np_state_dict[f"{attn_prefix}.proj.bias"],
                    )

                # MLP.
                elif isinstance(inner_layer, tf.keras.Model):
                    mlp_prefix = f"{common_prefix}.{block_idx}.mlp"
                    for mlp_layer in inner_layer.layers:
                        if isinstance(mlp_layer, tf.keras.layers.Dense):
                            mlp_layer = helpers.modify_tf_block(
                                mlp_layer,
                                np_state_dict[
                                    f"{mlp_prefix}.fc{mlp_layer_idx}.weight"
                                ],
                                np_state_dict[
                                    f"{mlp_prefix}.fc{mlp_layer_idx}.bias"
                                ],
                            )
                            mlp_layer_idx += 1

            block_idx += 1
    return tf_block


def main(args):
    if args.pre_logits:
        print(f"Converting {args.model_name} for feature extraction...")
    else:
        print(f"Converting {args.model_name}...")

    print("Instantiating PyTorch model...")
    pt_model = timm.create_model(model_name=args.model_name, pretrained=True)
    pt_model.eval()

    print("Instantiating TF model...")
    cfg_method = model_configs.MODEL_MAP[args.model_name]
    cfg = cfg_method()
    tf_model = SwinTransformer(**cfg, pre_logits=args.pre_logits)

    image_size = cfg.get("img_size", 224)
    dummy_inputs = tf.ones((2, image_size, image_size, 3))
    _ = tf_model(dummy_inputs)

    if not args.pre_logits:
        assert tf_model.count_params() == sum(
            p.numel() for p in pt_model.parameters()
        )

    # Load the PT params.
    np_state_dict = pt_model.state_dict()
    np_state_dict = {k: np_state_dict[k].numpy() for k in np_state_dict}

    print("Beginning parameter porting process...")

    # Projection.
    tf_model.projection.layers[0] = helpers.modify_tf_block(
        tf_model.projection.layers[0],
        np_state_dict["patch_embed.proj.weight"],
        np_state_dict["patch_embed.proj.bias"],
    )
    tf_model.projection.layers[2] = helpers.modify_tf_block(
        tf_model.projection.layers[2],
        np_state_dict["patch_embed.norm.weight"],
        np_state_dict["patch_embed.norm.bias"],
    )

    # Layer norm layers.
    ln_idx = -2
    tf_model.layers[ln_idx] = helpers.modify_tf_block(
        tf_model.layers[ln_idx],
        np_state_dict["norm.weight"],
        np_state_dict["norm.bias"],
    )

    # Head layers.
    if not args.pre_logits:
        head_layer = tf_model.get_layer("classification_head")
        tf_model.layers[-1] = helpers.modify_tf_block(
            head_layer,
            np_state_dict["head.weight"],
            np_state_dict["head.bias"],
        )

    # Swin layers.
    for i in range(len(cfg["depths"])):
        _ = modify_swin_blocks(
            np_state_dict,
            f"layers.{i}",
            tf_model.layers[i + 2].layers,
        )

    print("Porting successful, serializing TensorFlow model...")
    save_path = os.path.join(TF_MODEL_ROOT, args.model_name)
    save_path = f"{save_path}_fe" if args.pre_logits else save_path
    tf_model.save(save_path)
    print(f"TensorFlow model serialized to: {save_path}...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
