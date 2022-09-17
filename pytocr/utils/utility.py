import numpy as np
import cv2

def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def sort_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
            (_boxes[i+1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i+1]
            _boxes[i+1] = tmp
    return _boxes


def get_part_img(img, pts):
    '''
    pts:文本框顶点集 shape:(4, 2)
    '''
    pts = pts.astype(np.float32)
    left = int(np.min(pts[:, 0]))
    right = int(np.max(pts[:, 0]))
    top = int(np.min(pts[:, 1]))
    bottom = int(np.max(pts[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    pts -= np.array([left,top], dtype=np.float32)

    img_crop_w = int(right - left)
    img_crop_h = int(bottom - top)
    dst_pts = np.array([
        [0, 0], 
        [img_crop_w-1, 0], 
        [img_crop_w-1, img_crop_h-1], 
        [0, img_crop_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    rec_img = cv2.warpPerspective(
        img_crop, M, (img_crop_w, img_crop_h), 
        borderMode=cv2.BORDER_REPLICATE, 
        flags=cv2.INTER_LINEAR)

    return rec_img


def get_affine_transform(center, img_maxsize, target_size, inv=0):
    """
    功能是获取图像padding_resize到指定图像尺寸的仿射矩阵以及paddin_resize后的图像的坐标映射到原图像的仿射矩阵
    center：原图像的中心点坐标(cx,cy)
    img_maxsize：原图像的最长边
    target_size: 输出尺寸
    inv: 为0时计算的是正向的仿射变换矩阵
    仿射变换矩阵通过三个点计算得到
    第一个点为中心点，第二个点为中心点下方的边界点，第三个点通过对应原点的点（paddding后方图的左上角）
    """
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array((0, img_maxsize / 2.0))
    dst[0, :] = np.array((target_size / 2.0, target_size / 2.0))
    dst[1, :] = dst[0, :] + np.array((0, target_size / 2.0))
	#计算src第三个点
    if(center[0] >= center[1]):#代表原图的宽度大于高度
        src[2, :] = np.array((0, center[1] - center[0]))
    else:#代表原图的宽度小于高度
        src[2, :] = np.array((center[0] - center[1], 0))
	#计算仿射变换矩阵
    if inv:
        affineMat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        affineMat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return affineMat

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, img_maxsize, target_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, img_maxsize, target_size, inv=1)
    #print("后处理恢复到原图的仿射变换矩阵是",trans)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords 