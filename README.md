# Swin for win!

[![TensorFlow 2.8](https://img.shields.io/badge/TensorFlow-2.8-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0)
[![Models on TF-Hub](https://img.shields.io/badge/TF--Hub-Models%20on%20TF--Hub-orange)](https://tfhub.dev/sayakpaul/collections/swin/1)

This repository provides TensorFlow / Keras implementations of different Swin Transformer
[1, 2] variants by Liu et al. and Chen et al. It also provides the TensorFlow / Keras models
that have been populated with the original Swin pre-trained params available from [3, 4]. These
models are not blackbox SavedModels i.e., they can be fully expanded into `tf.keras.Model`
objects and one can call all the utility functions on them (example: `.summary()`).

Refer to the ["Using the models"](https://github.com/sayakpaul/swin-transformers-tf#using-the-models)
section to get started. 

I find Swin Transformers interesting because they induce a sense of hierarchies by using ***s***hifted ***win***dows. Multi-scale
representations like that are crucial to get good performance in tasks like object detection and segmentation.
![teaser](https://github.com/microsoft/Swin-Transformer/raw/main/figures/teaser.png)
<sup><a href=https://github.com/microsoft/Swin-Transformer>Source</a></sup>

"Swin for win!" however doesn't portray my architecture bias -- I found it cool and hence kept it.

## Table of contents

* [Conversion](https://github.com/sayakpaul/swin-transformers-tf#conversion)
* [Collection of pre-trained models (converted from PyTorch to TensorFlow)](https://github.com/sayakpaul/swin-transformers-tf#models)
* [Results of the converted models](https://github.com/sayakpaul/swin-transformers-tf#results)
* [How to use the models?](https://github.com/sayakpaul/swin-transformers-tf#using-the-models)
* [References](https://github.com/sayakpaul/swin-transformers-tf#references)
* [Acknowledgements](https://github.com/sayakpaul/swin-transformers-tf#acknowledgements)

## Conversion

TensorFlow / Keras implementations are available in `swins/models.py`. All model configurations
are in `swins/model_configs.py`. Conversion utilities are in `convert.py`. To run the conversion 
utilities, first install all the dependencies listed in `requirements.txt`. Additionally,
nnstall `timm` from source:

```sh
pip install -q git+https://github.com/rwightman/pytorch-image-models
```

## Models

Find the models on TF-Hub here: https://tfhub.dev/sayakpaul/collections/swin/1. You can fully inspect the
architecture of the TF-Hub models like so:

```py
import tensorflow as tf

model_gcs_path = "gs://tfhub-modules/sayakpaul/swin_tiny_patch4_window7_224/1/uncompressed"
model = tf.keras.models.load_model(model_gcs_path)

dummy_inputs = tf.ones((2, 224, 224, 3))
_ = model(dummy_inputs)
print(model.summary(expand_nested=True))
```

## Results

The table below provides a performance summary (ImageNet-1k validation set):

| model_name                     |   top1_acc(%) |   top5_acc(%) |   orig_top1_acc(%) |
|:------------------------------:|:-------------:|:-------------:|:------------------:|
| swin_base_patch4_window7_224   |        85.134 |        97.48  |               85.2 |
| swin_large_patch4_window7_224  |        86.252 |        97.878 |               86.3 |
| swin_s3_base_224               |        83.958 |        96.532 |               84   |
| swin_s3_small_224              |        83.648 |        96.358 |               83.7 |
| swin_s3_tiny_224               |        82.034 |        95.864 |               82.1 |
| swin_small_patch4_window7_224  |        83.178 |        96.24  |               83.2 |
| swin_tiny_patch4_window7_224   |        81.184 |        95.512 |               81.2 |
| swin_base_patch4_window12_384  |        86.428 |        98.042 |               86.4 |
| swin_large_patch4_window12_384 |        87.272 |        98.242 |               87.3 |


The `base` and `large` models were first pre-trained on the ImageNet-22k dataset and then fine-tuned
on the ImageNet-1k dataset.

[`in1k-eval` directory](https://github.com/sayakpaul/swin-transformers-tf/tree/main/in1k-eval) provides details
on how these numbers were generated. Original scores for all the models except for the `s3` ones were
gathered from [here](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md). Scores
for the `s3` model were gathered from [here](https://github.com/microsoft/Cream/tree/main/AutoFormerV2#model-zoo).

## Using the models

**Pre-trained models**:

* Off-the-shelf classification: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/swin-transformers-tf/blob/main/notebooks/classification.ipynb)
* Fine-tuning: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/swin-transformers-tf/blob/main/notebooks/finetune.ipynb)

When doing transfer learning try using the models that were pre-trained on the ImageNet-22k dataset. All the
`base` and `large` models listed here were pre-trained on the ImageNet-22k dataset. Refer to the
[model collection page on TF-Hub](https://tfhub.dev/sayakpaul/collections/swin/1) to know more.

These models also output attention weights from each of the Transformer blocks.
Refer to [this notebook](https://colab.research.google.com/github/sayakpaul/swin-transformers-tf/blob/main/notebooks/classification.ipynb)
for more details. Additionally, the notebook shows how to obtain the attention maps for a given image.

 
**Randomly initialized models**:
 
```py
import tensorflow as tf

from swins import SwinTransformer

cfg = dict(
    patch_size=4,
    window_size=7,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
)
 
swin_base_patch4_window7_224 = SwinTransformer(
    name="swin_base_patch4_window7_224", **cfg
)
print("Model instantiated, attempting predictions...")
random_tensor = tf.random.normal((2, 224, 224, 3))
outputs = swin_base_patch4_window7_224(random_tensor, training=False)

print(outputs.shape)

print(swin_base_patch4_window7_224.count_params() / 1e6)
```

To initialize a network with say, 5 classes do:

```py
cfg = dict(
    patch_size=4,
    window_size=7,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    num_classes=5,
)

swin_base_patch4_window7_224 = SwinTransformer(
    name="swin_base_patch4_window7_224", **cfg
)
```

To view different model configurations, refer to `swins/model_configs.py`.

## References

[1] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows Liu et al.](https://arxiv.org/abs/2103.14030)

[2] [Searching the Search Space of Vision Transformer by Chen et al.](https://arxiv.org/abs/2111.14725)

[3] [Swin Transformers GitHub](https://github.com/microsoft/Swin-Transformer)

[4] [AutoFormerV2 GitHub](https://github.com/silent-chen/AutoFormerV2-model-zoo)

## Acknowledgements

* [`timm` library source code](https://github.com/rwightman/pytorch-image-models)
for the awesome codebase. I've copy-pasted and modified a huge chunk of code from there.
I've also mentioned it from the respective scripts.
* [Willi Gierke](https://ch.linkedin.com/in/willi-gierke) for helping with a non-trivial model serialization hack.
* [ML-GDE program](https://developers.google.com/programs/experts/) for
providing GCP credits that supported my experiments.
