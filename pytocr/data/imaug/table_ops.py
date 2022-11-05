import cv2
import numpy as np


class ResizeTableImage(object):
    """
    固定长边, 短边等比例缩放, 可选Padding
    """
    def __init__(
        self, 
        max_len, 
        use_padding=False, 
        **kwargs
    ):
        super(ResizeTableImage, self).__init__()
        self.max_len = max(int(round(max_len / 32) * 32), 32)  # 32的整数倍
        self.use_padding = use_padding

    def __call__(self, data):
        img = data["image"]
        src_h, src_w = img.shape[:2]
        ratio = self.max_len / (max(src_h, src_w) * 1.0)
        resize_h = src_h * ratio
        resize_w = src_w * ratio
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        resize_img = cv2.resize(img, (resize_w, resize_h))
        data["image"] = resize_img
        data["shape"] = np.array([src_h, src_w, ratio, ratio, resize_h, resize_w])
        if self.use_padding:
            max_resize_len = max(resize_h, resize_w)
            pad_h, pad_w = max_resize_len, max_resize_len
            padding_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
            padding_img[0:resize_h, 0:resize_w, :] = resize_img
            data["image"] = padding_img
            data["shape"] = np.array([src_h, src_w, ratio, ratio, pad_h, pad_w])
        return data