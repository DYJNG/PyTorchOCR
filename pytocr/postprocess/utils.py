import numpy as np
import cv2

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