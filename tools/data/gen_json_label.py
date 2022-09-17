import numpy as np
import cv2
from PIL import Image
import random
from pathlib import Path
import argparse
import json
from tqdm import tqdm

random.seed(2022)

def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_boxes(img_path, label_path, sort_pts=False):
    boxes, txts = [], []
    if sort_pts:
        img_h, img_w = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]
    with open(label_path, "r", encoding="utf-8") as fid:
        lines = fid.readlines()
        for line in lines:
            line = line.replace("\ufeff", "").replace(
                "\xef\xbb\xbf", "").strip("\n")
            label = line.split(",")
            box = [int(round(float(pt), 0)) for pt in label[:8]]
            if len(box) == 8 and sort_pts:
                box = cv2.minAreaRect(np.array(box, dtype=np.int32).reshape(-1, 2))
                box = cv2.boxPoints(box)
                box = order_points_clockwise(
                    np.array(box, dtype=np.float32).reshape(-1, 2))
                box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
                box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
                box = box.astype(np.int32).tolist()
            else:
                box = np.array(box, dtype=np.int32).reshape(-1, 2).tolist()
            txt = "".join(label[8:])
            boxes.append(box)
            txts.append(txt)
    return boxes, txts


def check_img(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            img = Image.open(img_path).convert("RGB")
            img.save(img_path)
        except:
            print(img_path + " is a bad image")
            return False
    return True

def write_to_file_det(img_paths, label_dir, out_path, delimiter="\t", sort_pts=False):
    fp = open(str(out_path), "w", encoding="UTF-8")
    for img_path in tqdm(img_paths):
        if "dir_name:" not in str(label_dir):
            label_path = Path(str(label_dir)).joinpath(str(img_path.stem) + ".txt")
        else:
            label_path = Path(str(img_path.parent.parent.joinpath(
                str(label_dir).split(":")[-1].strip()))).joinpath(
                    str(img_path.stem) + ".txt")
        if (not check_img(str(img_path))) or (
            not label_path.is_file()): # not label_path.exists():
            print("img_path " + str(img_path))
            print("can not find " + str(label_path))
            continue
        boxes, txts = get_boxes(str(img_path), str(label_path), sort_pts)
        label = []
        for box, txt in zip(boxes, txts):
            label.append({"transcription": txt, "points": box})
        fp.write(str(img_path) + delimiter + json.dumps(
                label, ensure_ascii=False) + "\n")
    fp.close()


def gen_det_label(img_dir, label_dir, out_path, delimiter="\t", 
                  split_train_val=False, ratio=0.9, extra_out_path=None, 
                  sort_pts=False):
    img_dir = Path(str(img_dir))
    img_paths = []
    # for img_path in img_dir.glob("*.[jp][pn]g"):
    for img_path in img_dir.rglob("*.[jp][pn]g"):
        img_paths.append(img_path)
    if not split_train_val:
        write_to_file_det(img_paths, label_dir, out_path, delimiter, sort_pts)
    else:
        train_paths, val_paths = [], []
        num_imgs = len(img_paths)
        random.shuffle(img_paths)
        for i in range(num_imgs):
            if i < num_imgs * ratio:
                train_paths.append(img_paths[i])
            else:
                val_paths.append(img_paths[i])
        write_to_file_det(train_paths, label_dir, out_path, delimiter, sort_pts)
        write_to_file_det(val_paths, label_dir, extra_out_path, delimiter, sort_pts)
        print("num of train samples: ", len(train_paths))
        print("num of valid samples: ", len(val_paths))


def write_to_file_rec(img_paths, label_dir, out_path, delimiter="\t"):
    fp = open(str(out_path), "w", encoding="UTF-8")
    for img_path in tqdm(img_paths):
        if "dir_name:" not in str(label_dir):
            label_path = Path(str(label_dir)).joinpath(str(img_path.stem) + ".txt")
        else:
            label_path = Path(str(img_path.parent.parent.joinpath(
                str(label_dir).split(":")[-1].strip()))).joinpath(
                    str(img_path.stem) + ".txt")
        if (not check_img(str(img_path))) or (
            not label_path.is_file()): # not label_path.exists():
            print("can not find " + str(label_path))
            continue
        with open(str(label_path), "r", encoding="UTF-8") as f:
            label = f.readline().strip("\n").replace(" ", "") # remove space
        fp.write(str(img_path) + delimiter + label + "\n")
    fp.close()

def gen_rec_label(img_dir, label_dir, out_path, delimiter="\t",
                  split_train_val=False, ratio=0.9, extra_out_path=None):
    img_dir = Path(str(img_dir))
    img_paths = []
    for img_path in img_dir.rglob("*.[jp][pn]g"):
        img_paths.append(img_path)
    if not split_train_val:
        write_to_file_rec(img_paths, label_dir, out_path, delimiter)
    else:
        train_paths, val_paths = [], []
        num_imgs = len(img_paths)
        random.shuffle(img_paths)
        for i in range(num_imgs):
            if i < num_imgs * ratio:
                train_paths.append(img_paths[i])
            else:
                val_paths.append(img_paths[i])
        write_to_file_rec(train_paths, label_dir, out_path, delimiter)
        write_to_file_rec(val_paths, label_dir, extra_out_path, delimiter)
        print("num of train samples: ", len(train_paths))
        print("num of valid samples: ", len(val_paths))
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="det",
        help="Generate rec_label or det_label, can be set rec or det")
    parser.add_argument(
        "--img_dir",
        type=str,
        default=".",
        help="The directory of images")
    parser.add_argument(
        "--label_dir",
        type=str,
        default="dir_name:gt",  # label_dir path or "dir_name:[label_dir_name]"
        help="The directory of annotations")
    parser.add_argument(
        "--out_path",
        type=str,
        default="out_label.txt",
        help="Output file path")
    parser.add_argument(
        "--delimiter",
        type=str,
        default="\t",
        help="delimiter")
    parser.add_argument(
        "--sort_pts",
        type=bool,
        default=False,
        help="Whether sort box points")
    parser.add_argument(
        "--split_train_val",
        type=bool,
        default=False,
        help="Whether split train val")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="Ratio of train samples")
    parser.add_argument(
        "--extra_out_path",
        type=str,
        default="val_label.txt",
        help="Output val file path")


    args = parser.parse_args()
    if args.mode == "det":
        print("Generate det label")
        gen_det_label(args.img_dir, args.label_dir, args.out_path, args.delimiter, 
                      args.split_train_val, args.ratio, args.extra_out_path, args.sort_pts)
    elif args.mode == "rec":
        print("Generate rec label")
        gen_rec_label(args.img_dir, args.label_dir, args.out_path, args.delimiter, 
                      args.split_train_val, args.ratio, args.extra_out_path)