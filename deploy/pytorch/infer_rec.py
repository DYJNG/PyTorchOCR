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
from utils import load_config, draw_rec_res


def parse_args():
    parser = argparse.ArgumentParser(
        description="pytocr det_model infer")
    parser.add_argument("--config", type=str, help="configuration file to use")
    parser.add_argument("--character_dict_path", type=str, default=None, help="character dict file to use")
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


class Recer(object):
    def __init__(
        self, 
        rec_cfg, 
        rec_ckpt, 
        character_dict_path=None,
        gpu_id=0
    ) -> None:
        # ----- recer init ----- #
        rec_cfg = load_config(rec_cfg)
        rec_cfg["Global"]["distributed"] = False
        if character_dict_path is not None:
            rec_cfg["Global"]["character_dict_path"] = character_dict_path
        # build post process
        rec_post_process_class = build_post_process(rec_cfg["PostProcess"], rec_cfg["Global"])
        # for rec algorithm
        char_num = len(getattr(rec_post_process_class, "character"))
        rec_cfg["Architecture"]["Head"]["out_channels"] = char_num
        recer = build_model(rec_cfg["Architecture"])
        # check if set use_gpu=True
        use_gpu = rec_cfg["Global"]["use_gpu"] and torch.cuda.is_available()
        rec_device = torch.device("cuda:{}".format(gpu_id) if use_gpu else "cpu")
        recer = recer.to(rec_device)
        recer.eval()
        recer = load_pretrained_params(recer, rec_ckpt)
        # create data ops
        rec_transforms = []
        rec_img_mode = "GRAY"
        for op in rec_cfg["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "DecodeImage" in op_name:
                rec_img_mode = op[op_name]["img_mode"]
                continue
            elif "Label" in op_name:
                continue
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            rec_transforms.append(op)
        rec_ops = create_operators(rec_transforms, rec_cfg["Global"])
        self.recer = recer
        self.rec_post_process_class = rec_post_process_class
        self.rec_ops = rec_ops
        self.rec_device = rec_device
        self.rec_img_mode = rec_img_mode
    
    @torch.no_grad()
    def run(self, img_path):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        # ----- do rec -----#
        if self.rec_img_mode == "GRAY":
            rec_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.rec_img_mode == "RGB":
            rec_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rec_img = img.copy()
        rec_data = {"image": rec_img}
        rec_batch = transform(rec_data, self.rec_ops)
        rec_img = rec_batch[0].unsqueeze(dim=0) # image
        rec_img = rec_img.to(self.rec_device)
        rec_preds = self.recer(rec_img)
        rec_post_result = self.rec_post_process_class(rec_preds)
        # parser text and prob
        text, prob_rec = rec_post_result[0]
        prob_rec = round(prob_rec, 2)

        return text, prob_rec


@torch.no_grad()
def main():
    args = parse_args()
    myRecer = Recer(
        args.config, 
        args.model_path, 
        args.character_dict_path, 
        args.gpu_id
    )

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
        text, prob = myRecer.run(str(img_path))

        # save2txt
        save_txt_path = out_dir.joinpath("res_"+str(img_path.stem)+".txt")
        with open(str(save_txt_path), "w", encoding="UTF-8") as fp:
            fp.write(text + "," + str(prob) + "\n")

        # save2img
        save_img_path = out_dir.joinpath("res_"+str(img_path.stem)+".jpg")
        res_img = draw_rec_res(text, prob, str(img_path), str(save_img_path))
        if args.show: 
            cv2.imshow("rec_res", res_img)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()
