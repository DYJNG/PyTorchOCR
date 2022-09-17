import os
import yaml
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    global_config = AttrDict()
    default_config = {"Global": {"debug": False, }}  # 新建字典，避免覆盖global_config

    merge_config(default_config, global_config)
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, "rb"), Loader=yaml.Loader), global_config)
    return global_config


def merge_config(config, global_config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split(".")
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
                    

def draw_det_res(dt_boxes, img_path, save_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if len(dt_boxes) > 0:
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    cv2.imwrite(save_path, img)
    print("The detected Image saved in {}".format(save_path))
    return img


def draw_rec_res(text, prob, img_path, save_path):
    pilimg = Image.open(str(img_path)).convert("RGB")
    w, h = pilimg.size
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font_size = int(max(min(30, h-5), 10))
    font = ImageFont.truetype("fs_GB2312.ttf", font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小  显示汉字
    draw.text((2, 2), text + "," + str(prob), (0, 0, 255), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    pilimg.save(save_path)

    img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    print("The Rec_res Image saved in {}".format(save_path))
    return img


def draw_cls_res(pred_cls, prob, img_path, save_path):
    pilimg = Image.open(str(img_path)).convert("RGB")
    w, h = pilimg.size
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font_size = int(max(min(30, h-5), 10))
    font = ImageFont.truetype("fs_GB2312.ttf", font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小  显示汉字
    draw.text((2, 2), pred_cls + "," + str(prob), (0, 0, 255), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    pilimg.save(save_path)

    img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    print("The Rec_res Image saved in {}".format(save_path))
    return img


def draw_ocr_res(ocr_res, img_path, save_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if len(ocr_res) > 0:
        for cur_res in ocr_res:
            ori_box, text, prob = cur_res
            box = ori_box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
            pilimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            h = min(cv2.minAreaRect(box.reshape((-1, 2)))[1])
            font_size = int(max(min(30, h-5), 10))
            font = ImageFont.truetype("fs_GB2312.ttf", font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
            draw.text((ori_box[0][0], max(0, ori_box[0][1] - 10)), text + "," + str(prob), (0, 0, 255), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
            img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)
    print("The OCR_res Image saved in {}".format(save_path))
    return img