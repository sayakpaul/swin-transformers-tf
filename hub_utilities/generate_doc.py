"""Generates model documentation for Swin-TF models.

Credits: Willi Gierke
"""

import os
from string import Template

import attr

template = Template(
    """# Module $HANDLE

Fine-tunable Swin Transformer model pre-trained on the $DATASET_DESCRIPTION.

<!-- asset-path: https://storage.googleapis.com/swin-tf/tars/$ARCHIVE_NAME.tar.gz  -->
<!-- task: image-classification -->
<!-- network-architecture: swin-transformer -->
<!-- format: saved_model_2 -->
<!-- fine-tunable: true -->
<!-- license: mit -->
<!-- colab: https://colab.research.google.com/github/sayakpaul/swin-transformers-tf/blob/main/notebooks/finetune.ipynb -->

## Overview

This model is a Swin Transformer [1] pre-trained on the $DATASET_DESCRIPTION. You can find the complete
collection of Swin models on TF-Hub on [this page](https://tfhub.dev/sayakpaul/collections/swin/1).

You can use this model for feature extraction and fine-tuning. Please refer to
the Colab Notebook linked on this page for more details.

## Notes

* The original model weights are provided from [2]. There were ported to Keras models
(`tf.keras.Model`) and then serialized as TensorFlow SavedModels. The porting
steps are available in [3].
* If the model handle contains `s3` then please refer to [4] for more details on the architecture. It's 
original weights are available in [5].
* The model can be unrolled into a standard Keras model and you can inspect its topology.
To do so, first download the model from TF-Hub and then load it using `tf.keras.models.load_model`
providing the path to the downloaded model folder.

## References

[1] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows Liu et al.](https://arxiv.org/abs/2103.14030)

[2] [Swin Transformers GitHub](https://github.com/microsoft/Swin-Transformer)

[3] [Swin-TF GitHub](https://github.com/sayakpaul/swin-transformers-tf)

[4] [Searching the Search Space of Vision Transformer by Chen et al.](https://arxiv.org/abs/2111.14725)

[5] [AutoFormerV2 GitHub](https://github.com/silent-chen/AutoFormerV2-model-zoo)

## Acknowledgements

* [Willi](https://ch.linkedin.com/in/willi-gierke)
* [ML-GDE program](https://developers.google.com/programs/experts/)

"""
)


@attr.s
class Config:
    size = attr.ib(type=str)
    patch_size = attr.ib(type=int)
    window_size = attr.ib(type=int)
    single_resolution = attr.ib(type=int)
    dataset = attr.ib(type=str)
    type = attr.ib(type=str, default="swin")

    def two_d_resolution(self):
        return f"{self.single_resolution}x{self.single_resolution}"

    def gcs_folder_name(self):
        if self.dataset == "in22k":
            return f"swin_{self.size}_patch{self.patch_size}_window{self.window_size}_{self.single_resolution}_{self.dataset}_fe"
        elif self.type == "autoformer":
            return f"swin_s3_{self.size}_{self.single_resolution}_fe"
        else:
            return f"swin_{self.size}_patch{self.patch_size}_window{self.window_size}_{self.single_resolution}_fe"

    def handle(self):
        return f"sayakpaul/{self.gcs_folder_name()}/1"

    def rel_doc_file_path(self):
        """Relative to the tfhub.dev directory."""
        return f"assets/docs/{self.handle()}.md"


# swin_base_patch4_window12_384, swin_base_patch4_window12_384_in22k
for c in [
    Config("tiny", 4, 7, 224, "in1k"),
    Config("small", 4, 7, 224, "in1k"),
    Config("base", 4, 7, 224, "in1k"),
    Config("base", 4, 12, 384, "in1k"),
    Config("large", 4, 7, 224, "in1k"),
    Config("large", 4, 12, 384, "in1k"),
    Config("base", 4, 7, 224, "in22k"),
    Config("base", 4, 12, 384, "in22k"),
    Config("large", 4, 7, 224, "in22k"),
    Config("large", 4, 12, 384, "in22k"),
    Config("tiny", 0, 0, 224, "in1k", "autoformer"),
    Config("small", 0, 0, 224, "in1k", "autoformer"),
    Config("base", 0, 0, 224, "in1k", "autoformer"),
]:
    if c.dataset == "in1k" and not ("large" in c.size or "base" in c.size):
        dataset_text = "ImageNet-1k dataset"
    elif c.dataset == "in22k":
        dataset_text = "ImageNet-22k dataset"
    elif c.dataset == "in1k" and ("large" in c.size or "base" in c.size):
        dataset_text = (
            "ImageNet-22k"
            " dataset and"
            " was then "
            "fine-tuned "
            "on the "
            "ImageNet-1k "
            "dataset"
        )

    save_path = os.path.join(
        "/Users/sayakpaul/Downloads/", "tfhub.dev", c.rel_doc_file_path()
    )
    model_folder = save_path.split("/")[-2]
    model_abs_path = "/".join(save_path.split("/")[:-1])

    if not os.path.exists(model_abs_path):
        os.makedirs(model_abs_path, exist_ok=True)

    with open(save_path, "w") as f:
        f.write(
            template.substitute(
                HANDLE=c.handle(),
                DATASET_DESCRIPTION=dataset_text,
                INPUT_RESOLUTION=c.two_d_resolution(),
                ARCHIVE_NAME=c.gcs_folder_name(),
            )
        )
