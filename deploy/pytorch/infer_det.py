import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import numpy as np
import cv2
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from pytocr.data import create_operators, transform
from pytocr.modeling.architectures import build_model
from pytocr.postprocess import build_post_process
from pytocr.utils.save_load import load_pretrained_params
from pytocr.utils.utility import sort_boxes
from utils import load_config, draw_det_res


def parse_args():
    parser = argparse.ArgumentParser(
        description="pytocr det_model infer")
    parser.add_argument("--config", type=str, help="configuration file to use")
    parser.add_argument("--model_path", type=str, help="model weights file to use")
    parser.add_argument("--img_path", type=str, help="test img-path or img-dir")
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--out_dir", type=str, help="directory where painted images will be saved")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="id of gpu to use "
        "(only applicable to non-distributed testing)")
    args = parser.parse_args()

    return args


class Deter(object):
    def __init__(
        self, 
        det_cfg, 
        det_ckpt, 
        gpu_id=0
    ) -> None:
        # ----- detecter init ----- #
        det_cfg = load_config(det_cfg)
        det_cfg["Global"]["distributed"] = False
        deter = build_model(det_cfg["Architecture"])
        # check if set use_gpu=True
        use_gpu = det_cfg["Global"]["use_gpu"] and torch.cuda.is_available()
        det_device = torch.device("cuda:{}".format(gpu_id) if use_gpu else "cpu")
        deter = deter.to(det_device)
        deter.eval()
        deter = load_pretrained_params(deter, det_ckpt)
        # build post process
        det_post_process_class = build_post_process(det_cfg["PostProcess"], det_cfg["Global"])
        # create data ops
        det_transforms = []
        det_img_mode = "RGB"
        for op in det_cfg["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "DecodeImage" in op_name:
                det_img_mode = op[op_name]["img_mode"]
                continue
            elif "Label" in op_name:
                continue
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image", "shape"]
            det_transforms.append(op)
        det_ops = create_operators(det_transforms, det_cfg["Global"])
        self.deter = deter
        self.det_post_process_class = det_post_process_class
        self.det_ops = det_ops
        self.det_device = det_device
        self.det_img_mode = det_img_mode

    @torch.no_grad()
    def run(self, img_path):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        # ----- do det -----#
        if self.det_img_mode == "RGB":
            det_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            det_img = img.copy()
        det_data = {"image": det_img}
        det_batch = transform(det_data, self.det_ops)
        det_img = det_batch[0].unsqueeze(dim=0) # image
        det_shape_list = np.expand_dims(det_batch[1], axis=0)
        det_img = det_img.to(self.det_device)
        det_preds = self.deter(det_img)
        det_post_result = self.det_post_process_class(det_preds, det_shape_list)
        # parser boxes if post_result is dict
        boxes = sort_boxes(det_post_result[0]["points"])  # 由上到下 由左到右 排版

        return boxes


@torch.no_grad()
def main():
    args = parse_args()
    myDeter = Deter(
        args.config, 
        args.model_path, 
        args.gpu_id)

    assert os.path.exists(args.img_path), "img_path not exists"
    
    img_paths = []
    if os.path.isfile(args.img_path):
        img_paths.append(Path(args.img_path))
    elif os.path.isdir(args.img_path):
        for img_path in Path(args.img_path).glob("*.[jp][pn]g"):
            img_paths.append(img_path)
    
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
    else:
        out_dir = Path("./output")
        out_dir.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(img_paths):
        boxes = myDeter.run(str(img_path))

        # save2txt
        save_txt_path = out_dir.joinpath("res_"+str(img_path.stem)+".txt")
        with open(str(save_txt_path), "w", encoding="UTF-8") as fp:
            for box in boxes:
                box = [str(coor) for coor in box.reshape(-1).tolist()]
                fp.write(",".join(box) + "\n")
        
        # save2img
        save_img_path = out_dir.joinpath("res_"+str(img_path.stem)+".jpg")
        res_img = draw_det_res(boxes, str(img_path), str(save_img_path))
        if args.show: 
            cv2.imshow("det_res", res_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
