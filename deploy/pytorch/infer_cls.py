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
from utils import load_config, draw_cls_res


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


class Clser(object):
    def __init__(
        self,  
        cls_cfg=None, 
        cls_ckpt=None,
        gpu_id=0
    ) -> None:
        # ----- clser init ----- #
        cls_cfg = load_config(cls_cfg)
        cls_cfg["Global"]["distributed"] = False
        clser = build_model(cls_cfg["Architecture"])
        # check if set use_gpu=True
        use_gpu = cls_cfg["Global"]["use_gpu"] and torch.cuda.is_available()
        cls_device = torch.device("cuda:{}".format(gpu_id) if use_gpu else "cpu")
        clser = clser.to(cls_device)
        clser.eval()
        clser = load_pretrained_params(clser, cls_ckpt)
        # build post process
        cls_post_process_class = build_post_process(cls_cfg["PostProcess"], cls_cfg["Global"])
        # create data ops
        cls_transforms = []
        cls_img_mode = "RGB"
        for op in cls_cfg["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "DecodeImage" in op_name:
                cls_img_mode = op[op_name]["img_mode"]
                continue
            elif "Label" in op_name:
                continue
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            cls_transforms.append(op)
        cls_ops = create_operators(cls_transforms, cls_cfg["Global"])
        self.clser = clser
        self.cls_post_process_class = cls_post_process_class
        self.cls_ops = cls_ops
        self.cls_device = cls_device
        self.cls_img_mode = cls_img_mode
    
    @torch.no_grad()
    def run(self, img_path):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        # ----- do cls -----#
        if self.cls_img_mode == "GRAY":
            cls_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.cls_img_mode == "RGB":
            cls_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            cls_img = img.copy()
        cls_data = {"image": cls_img}
        cls_batch = transform(cls_data, self.cls_ops)
        cls_img = cls_batch[0].unsqueeze(dim=0) # image
        cls_img = cls_img.to(self.cls_device)
        cls_preds = self.clser(cls_img)
        cls_post_result = self.cls_post_process_class(cls_preds)
        # parser text and prob
        pred_cls, prob_cls = cls_post_result[0]
        prob_cls = round(prob_cls, 2)
            
        return pred_cls, prob_cls


@torch.no_grad()
def main():
    args = parse_args()
    myClser = Clser(
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
        pred_cls, prob = myClser.run(str(img_path))

        # save2txt
        save_txt_path = out_dir.joinpath("res_"+str(img_path.stem)+".txt")
        with open(str(save_txt_path), "w", encoding="UTF-8") as fp:
            fp.write(pred_cls + "," + str(prob) + "\n")

        save_img_path = out_dir.joinpath("res_"+str(img_path.stem)+".jpg")
        res_img = draw_cls_res(pred_cls, prob, str(img_path), str(save_img_path))
        if args.show: 
            cv2.imshow("cls_res", res_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
