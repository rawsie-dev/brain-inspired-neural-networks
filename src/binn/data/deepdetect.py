import os
from .vision import get_imagefolder


def get_deepdetect(seed, batch_size=32, data_path=None, image_size=224, num_workers=4):
    root = data_path or os.getenv("DEEPDETECT_DIR")

    if not root:
        import kagglehub
        root = os.path.join(kagglehub.dataset_download("ayushmandatta1/deepdetect-2025"), "ddata")

    return get_imagefolder(seed, os.path.join(root, "train"), os.path.join(root, "test"), batch_size, image_size, num_workers)
