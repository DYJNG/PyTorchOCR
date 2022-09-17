import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter as pyt_ColorJitter

__all__  = ["ColorJitter"]

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, **kwargs):
        self.aug = pyt_ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data):
        img = data["image"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        assert isinstance(img, 
                Image.Image), "'img' should convert to PIL.Image format"
        img = self.aug(img)
        data["image"] = np.asarray(img)
        return data
