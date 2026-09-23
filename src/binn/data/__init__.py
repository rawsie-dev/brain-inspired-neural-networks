from .adult import get_adult
from .drybean import get_drybean
from .deepdetect import get_deepdetect
from .tinyimagenet import get_tiny_imagenet


DATASETS = {"adult": get_adult, "drybean": get_drybean, "deepdetect": get_deepdetect, "tinyimagenet": get_tiny_imagenet}
