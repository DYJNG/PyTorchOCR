import cv2
import numpy as np
import torch
import pyclipper
from shapely.geometry import Polygon

from pytocr.utils.utility import transform_preds
from .db_postprocess_fast import cpp_boxes_from_bitmap

class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.5,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="poly",
                 cpp_speedup=False,
                 out_polygon=False, 
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.out_polygon = out_polygon
        self.score_mode = score_mode
        assert score_mode in [
            "box", "poly"
        ], "Score mode must be in [box, poly] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

        self.cpp_speedup = cpp_speedup
        
    def __call__(self, outs_dict, shape_list, 
                 use_padding_resize=False):
        pred = outs_dict["maps"]
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        res_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            src_h, src_w = int(src_h), int(src_w)
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            if self.cpp_speedup:
                tmp_boxes = cpp_boxes_from_bitmap(pred[batch_index], mask, 
                                                  self.box_thresh,
                                                  self.unclip_ratio, 
                                                  src_w, src_h, 
                                                  use_padding_resize)
                boxes = []
                scores = []
                for i in range(len(tmp_boxes)):
                    boxes.append(tmp_boxes[i])
                    scores.append(1.0)
                boxes = np.array(boxes, dtype=np.int16)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                    src_w, src_h, use_padding_resize)
            res_batch.append({"points": boxes, "scores": scores})
        return res_batch

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height, 
                          use_padding_resize=False):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """
        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            if self.out_polygon:
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape((-1, 2))
                if points.shape[0] < 4:
                    continue
            else:
                points, sside = self.get_mini_boxes(contour)
                if sside < self.min_size:
                    continue
                points = np.array(points).reshape((-1, 2))
            if self.score_mode == "box":
                score = self.box_score(pred, points)
            else:
                score = self.box_score(pred, contour.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            box = self.unclip(points)
            if len(box) != 1:
                continue
            box= box.reshape(-1, 1, 2)
            if self.out_polygon:
                _, sside = self.get_mini_boxes(box)
            else:
                box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box).reshape(-1, 2)
            if use_padding_resize:
                # 对预处理是padding_resize的方式的坐标rescale到原图像坐标
                center = np.array([dest_width / 2., dest_height / 2.], dtype=np.float32) # c为cx,cy
                src_maxsize = max(dest_width, dest_height) * 1.0                         # s为原图的最长边
                target_size = height                                                     # padding为方图，宽高都一样
                box = transform_preds(box, center, src_maxsize, target_size)
                box[:, 0] = np.clip(
                    np.round(box[:, 0]), 0, dest_width)
                box[:, 1] = np.clip(
                    np.round(box[:, 1]), 0, dest_height)
            else:
                box[:, 0] = np.clip(
                    np.round(box[:, 0] / width * dest_width), 0, dest_width)
                box[:, 1] = np.clip(
                    np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        boxes = np.array(boxes, dtype=np.int16)
        return boxes, scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score(self, bitmap, _pts):
        """
        box_score: use bbox/polyon mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        pts = _pts.copy()
        
        xmin = np.clip(np.floor(pts[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(pts[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(pts[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(pts[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        pts[:, 0] = pts[:, 0] - xmin
        pts[:, 1] = pts[:, 1] - ymin
        cv2.fillPoly(mask, pts.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    

class DistillationDBPostProcess(object):
    def __init__(self,
                 model_name=["student"],
                 key=None,
                 thresh=0.3,
                 box_thresh=0.5,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="poly",
                 cpp_speedup=False,
                 out_polygon=False, 
                 **kwargs):
        self.model_name = model_name
        self.key = key
        self.post_process = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode, 
            cpp_speedup=cpp_speedup,
            out_polygon=out_polygon)

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k], shape_list=shape_list)
        return results