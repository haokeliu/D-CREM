"""D-CREM data loading: tabular (reusing crem.data) and image (VOC/COCO)."""
from .tabular import TabularDataset, get_tabular_loaders, get_tabular_protocol
from .image import (
    ImageDataset, get_image_loaders, get_image_protocol,
    get_voc2007_raw_protocol,
)

__all__ = [
    "TabularDataset", "get_tabular_loaders", "get_tabular_protocol",
    "ImageDataset", "get_image_loaders", "get_image_protocol",
    "get_voc2007_raw_protocol",
]
