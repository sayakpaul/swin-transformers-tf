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
