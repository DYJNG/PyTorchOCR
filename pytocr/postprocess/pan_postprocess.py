import numpy as np
import cv2
import torch
from torch.nn import functional as F

from pytocr.postprocess.pan_postprocess_fast import pa
from pytocr.utils.utility import order_points_clockwise


class PANPostProcess(object):
    """
    The post process for PAN.
    """

    def __init__(self,
                 thresh=0.5,
                 box_thresh=0.85,
                 min_area=16,
                 min_kernel_area=2.6,
                 scale=4,
                 out_polygon=False, 
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.min_area = min_area
        self.min_kernel_area = min_kernel_area / float(scale ** 2)
        self.out_polygon = out_polygon
        self.scale = scale

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict["maps"]   # N * C * H * W
        assert isinstance(pred, torch.Tensor)
        self.img_h = pred.shape[2] * 4
        self.img_w = pred.shape[3] * 4
        # 上采样到预处理后的输入图像尺寸做后处理（更准），也可以在1/4, 1/2下采样尺寸做（更快）
        if self.scale != 4:
            pred = F.interpolate(
                pred, scale_factor=4 // self.scale, mode="nearest") # mode="bilinear"

        score = F.sigmoid(pred[:, 0, :, :])

        kernels = (pred[:, :2, :, :] > self.thresh).float()      # text + kernel batch * 2 * H * W
        text_mask = kernels[:, 0:1, :, :]      # text  batch * 1 * H * W   kernels[:, :1, :, :] != kernels[:, 0, :, :]
        kernels[:, 1:2, :, :] = kernels[:, 1:2, :, :] * text_mask   # kernel  batch * 1 * H * W

        emb = pred[:, 2:, :, :]
        emb = emb * text_mask.float()  # instance vector

        score = score.detach().cpu().numpy().astype(np.float32)
        kernels = kernels.detach().cpu().numpy().astype(np.uint8)
        emb = emb.detach().cpu().numpy().astype(np.float32)

        res_batch = []
        for batch_index in range(pred.shape[0]):
            boxes, scores = self.boxes_from_bitmap(score[batch_index], 
                                                   kernels[batch_index], 
                                                   emb[batch_index], 
                                                   shape_list[batch_index])
            res_batch.append({"points": boxes, "scores": scores})
        return res_batch

    def boxes_from_bitmap(self, score, kernels, emb, shape):
        label = pa(kernels, emb, self.min_kernel_area)  # int32
        # rescale to model_input_size
        if self.scale != 1:
            # int32 作为cv2的入参没问题吗？TODO：
            label = cv2.resize(label, (self.img_w, self.img_h),
                            interpolation=cv2.INTER_NEAREST)
            score = cv2.resize(score, (self.img_w, self.img_h),
                            interpolation=cv2.INTER_NEAREST)
        return self.generate_box(score, label, shape)

    def generate_box(self, score, label, shape):
        src_h, src_w, ratio_h, ratio_w = shape
        label_num = np.max(label) + 1

        boxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))[:, ::-1]  # ::-1 coor_h,coor_w -> coor_w,coor_h

            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < self.box_thresh:
                label[ind] = 0
                continue

            if not self.out_polygon:
                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect)
                bbox = order_points_clockwise(bbox)
            else:
                box_height = np.max(points[:, 1]) + 10
                box_width = np.max(points[:, 0]) + 10

                mask = np.zeros((box_height, box_width), dtype=np.uint8)
                mask[points[:, 1], points[:, 0]] = 255

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                bbox = np.squeeze(contours[0], 1)

            # 还原到原图尺寸
            bbox[:, 0] = np.clip(np.round(bbox[:, 0] / ratio_w), 0, src_w)
            bbox[:, 1] = np.clip(np.round(bbox[:, 1] / ratio_h), 0, src_h)
            boxes.append(bbox.astype(np.int16))
            scores.append(score_i)
        boxes = np.array(boxes, dtype=np.int16)
        return boxes, scores
