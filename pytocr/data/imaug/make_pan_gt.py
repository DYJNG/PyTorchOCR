import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

__all__ = ["MakePanGt"]


class MakePanGt(object):
    def __init__(self, size=640, min_shrink_ratio=0.5, **kwargs):
        self.min_shrink_ratio = min_shrink_ratio
        self.size = size

    def __call__(self, data):

        image = data["image"]
        text_polys = data["polys"]
        ignore_tags = data["ignore_tags"]

        h, w, _ = image.shape
        short_edge = min(h, w)
        if short_edge < self.size:
            # keep short_size >= self.size
            scale = self.size / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            text_polys *= scale

        gt_kernels = []
        for rate in [1.0, self.min_shrink_ratio]:   # text + kernel
            text_kernel, ignore_tags = self.generate_kernel(
                image.shape[0:2], rate, text_polys, ignore_tags)
            gt_kernels.append(text_kernel)

        gt_instance = np.zeros(image.shape[0:2], dtype=np.uint8) # cv2 入参只能是float(0-1) 或者uint8类型
        training_mask = np.ones(image.shape[0:2], dtype=np.uint8)
        for i in range(text_polys.shape[0]):
            cv2.fillPoly(gt_instance,
                         text_polys[i].astype(np.int32)[np.newaxis, :, :],
                         i + 1) # 0是背景，每个文本框是一类 TODO：uint8类型如果超过255个文本实例该怎么处理？
            if ignore_tags[i]:
                cv2.fillPoly(training_mask,
                             text_polys[i].astype(np.int32)[np.newaxis, :, :],
                             0)

        gt_kernels = np.array(gt_kernels, dtype=np.float32)
        gt_kernels[gt_kernels > 0] = 1    # 所有文本框都转换为一类

        data["image"] = image
        data["polys"] = text_polys
        data["gt_text"] = gt_kernels[0]     # text region  batch * H * W
        data["gt_kernels"] = gt_kernels[1]  # kernel       batch * H * W
        data["gt_instance"] = gt_instance.astype(np.int32)  # instance batch * H * W
        data["mask"] = training_mask.astype(np.float32)  # training_mask  batch * H * W
        return data

    def generate_kernel(self,
                        img_size,
                        shrink_ratio,
                        text_polys,
                        ignore_tags=None):
        """
        Refer to part of the code:
        https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/textdet_targets/base_textdet_targets.py
        """

        h, w = img_size
        text_kernel = np.zeros((h, w), dtype=np.uint8)
        for i, poly in enumerate(text_polys):
            polygon = Polygon(poly)
            distance = polygon.area * (1 - shrink_ratio * shrink_ratio) / (
                polygon.length + 1e-6)
            subject = [tuple(l) for l in poly]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked = np.array(pco.Execute(-distance))

            if len(shrinked) == 0 or shrinked.size == 0:
                if ignore_tags is not None:
                    ignore_tags[i] = True
                continue
            try:
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
            except:
                if ignore_tags is not None:
                    ignore_tags[i] = True
                continue
            # 若text_kernel是float类型，则填充数值最大只能为1 超过1则被截断为1，下行代码 i+1 永远等价于 1
            cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)], i + 1)  # 每个文本框是一类
        return text_kernel, ignore_tags
