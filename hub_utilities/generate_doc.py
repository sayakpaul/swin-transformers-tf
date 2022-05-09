"""Generates model documentation for Swin-TF models.

Credits: Willi Gierke
"""

import os
from string import Template

import attr

template = Template(
    """# Module $HANDLE

CaiT model pre-trained on the $DATASET_DESCRIPTION suitable for off-the-shelf classification.

<!-- asset-path: https://storage.googleapis.com/cait-tf/tars/$ARCHIVE_NAME.tar.gz  -->
<!-- task: image-classification -->
<!-- network-architecture: cait -->
<!-- format: saved_model_2 -->
<!-- fine-tunable: true -->
<!-- license: mit -->
<!-- colab: https://colab.research.google.com/github/sayakpaul/cait-tf/blob/main/notebooks/classification.ipynb -->

## Overview

This model is a CaiT [1] model pre-trained on the $DATASET_DESCRIPTION. You can find the complete
collection of CaiT models on TF-Hub on [this page](https://tfhub.dev/sayakpaul/collections/cait/1).

You can use this model for performing off-the-shelf classification. Please refer to
the Colab Notebook linked on this page for more details.

## Notes

* The original model weights are provided from [2]. There were ported to Keras models
(`tf.keras.Model`) and then serialized as TensorFlow SavedModels. The porting
steps are available in [3].
* The model can be unrolled into a standard Keras model and you can inspect its topology.
To do so, first download the model from TF-Hub and then load it using `tf.keras.models.load_model`
providing the path to the downloaded model folder.

## References

[1] [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)
[2] [CaiT GitHub](https://github.com/facebookresearch/deit)
[3] [CaiT-TF GitHub](https://github.com/sayakpaul/cait-tf)

## Acknowledgements

* [ML-GDE program](https://developers.google.com/programs/experts/)

"""
)


@attr.s
class Config:
    depth = attr.ib(type=str)
    single_resolution = attr.ib(type=int)

    def two_d_resolution(self):
        return f"{self.single_resolution}x{self.single_resolution}"

    def gcs_folder_name(self):
        return f"cait_{self.depth}_{self.single_resolution}"

    def handle(self):
        return f"sayakpaul/cait_{self.depth}_{self.single_resolution}/1"

    def rel_doc_file_path(self):
        """Relative to the tfhub.dev directory."""
        return f"assets/docs/{self.handle()}.md"


for c in [
    Config("xxs24", 224),
    Config("xxs24", 384),
    Config("xxs36", 224),
    Config("xxs36", 384),
    Config("xs24", 384),
    Config("s24", 224),
    Config("s24", 384),
    Config("s36", 384),
    Config("m36", 384),
    Config("m48", 448),
]:
    dataset_text = "ImageNet-1k dataset"

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
