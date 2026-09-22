import os
from .vision import get_imagefolder


def get_tiny_imagenet(seed, batch_size=32, data_path=None, image_size=224, num_workers=4):
    root = data_path or os.getenv("TINY_IMAGENET_DIR")

    if not root:
        import kagglehub
        root = os.path.join(kagglehub.dataset_download("akash2sharma/tiny-imagenet"), "tiny-imagenet-200")

    return get_imagefolder(seed, os.path.join(root, "train"), os.path.join(root, "val"), batch_size, image_size, num_workers, tiny_val=True)
