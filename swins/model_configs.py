# Take from here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py


def swin_base_patch4_window12_384():
    """Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window12_384",
    )
    return cfg


def swin_base_patch4_window7_224():
    """Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window7_224",
    )
    return cfg


def swin_large_patch4_window12_384():
    """Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window12_384",
    )
    return cfg


def swin_large_patch4_window7_224():
    """Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window7_224",
    )
    return cfg


def swin_small_patch4_window7_224():
    """Swin-S @ 224x224, trained ImageNet-1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_small_patch4_window7_224",
    )
    return cfg


def swin_tiny_patch4_window7_224():
    """Swin-T @ 224x224, trained ImageNet-1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_tiny_patch4_window7_224",
    )
    return cfg


def swin_base_patch4_window12_384_in22k():
    """Swin-B @ 384x384, trained ImageNet-22k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window12_384_in22k",
        num_classes=21841,
    )
    return cfg


def swin_base_patch4_window7_224_in22k():
    """Swin-B @ 224x224, trained ImageNet-22k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window7_224_in22k",
        num_classes=21841,
    )
    return cfg


def swin_large_patch4_window12_384_in22k():
    """Swin-L @ 384x384, trained ImageNet-22k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window12_384_in22k",
        num_classes=21841,
    )
    return cfg


def swin_large_patch4_window7_224_in22k():
    """Swin-L @ 224x224, trained ImageNet-22k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window7_224_in22k",
        num_classes=21841,
    )
    return cfg


def swin_s3_tiny_224():
    """Swin-S3-T @ 224x224, ImageNet-1k. https://arxiv.org/abs/2111.14725"""
    cfg = dict(
        patch_size=4,
        window_size=(7, 7, 14, 7),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_s3_tiny_224",
    )
    return cfg


def swin_s3_small_224():
    """Swin-S3-S @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725"""
    cfg = dict(
        patch_size=4,
        window_size=(14, 14, 14, 7),
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_s3_small_224",
    )
    return cfg


def swin_s3_base_224():
    """Swin-S3-B @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725"""
    cfg = dict(
        patch_size=4,
        window_size=(7, 7, 14, 7),
        embed_dim=96,
        depths=(2, 2, 30, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_s3_base_224",
    )
    return cfg


MODEL_MAP = {
    "swin_base_patch4_window12_384": swin_base_patch4_window12_384,
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    "swin_large_patch4_window12_384": swin_large_patch4_window12_384,
    "swin_large_patch4_window7_224": swin_large_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
    "swin_base_patch4_window12_384_in22k": swin_base_patch4_window12_384_in22k,
    "swin_base_patch4_window7_224_in22k": swin_base_patch4_window7_224_in22k,
    "swin_large_patch4_window12_384_in22k": swin_large_patch4_window12_384_in22k,
    "swin_large_patch4_window7_224_in22k": swin_large_patch4_window7_224_in22k,
    "swin_s3_tiny_224": swin_s3_tiny_224,
    "swin_s3_small_224": swin_s3_small_224,
    "swin_s3_base_224": swin_s3_base_224,
}
