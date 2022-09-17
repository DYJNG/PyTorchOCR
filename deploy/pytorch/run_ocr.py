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
from pytocr.utils.utility import sort_boxes, get_part_img
from utils import load_config, draw_ocr_res


def parse_args():
    parser = argparse.ArgumentParser(
        description="pytocr det_model infer")
    parser.add_argument("--det_config", type=str, help="det configuration file to use")
    parser.add_argument("--det_model_path", type=str, help="det model weights file to use")
    parser.add_argument("--rec_config", type=str, help="rec configuration file to use")
    parser.add_argument("--character_dict_path", type=str, default=None, help="character dict file to use")
    parser.add_argument("--rec_model_path", type=str, help="rec model weights file to use")
    parser.add_argument("--cls_config", type=str, default=None, help="cls configuration file to use")
    parser.add_argument("--cls_model_path", type=str, default=None, help="cls model weights file to use")
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


class OCRer(object):
    def __init__(
        self, 
        det_cfg, 
        det_ckpt, 
        rec_cfg, 
        rec_ckpt, 
        cls_cfg=None, 
        cls_ckpt=None,
        character_dict_path=None,
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
        # ----- clser init ----- #
        if cls_cfg is not None and cls_ckpt is not None:
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
        else:
            self.clser = None
            self.cls_post_process_class = None
            self.cls_ops = None
    
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
        boxes = sort_boxes(det_post_result[0]["points"])

        ocr_res = []
        # write result
        for box in boxes:
            part_img = get_part_img(img, box)
            part_img_h, part_img_w = part_img.shape[:2]
            if part_img_h >= 1.5 * part_img_w:
                part_img = np.rot90(part_img, 1) #逆时针旋转
            if self.clser is not None:
                # ----- do cls -----#
                if self.cls_img_mode == "GRAY":
                    cls_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)
                elif self.cls_img_mode == "RGB":
                    cls_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB)
                else:
                    cls_img = part_img.copy()
                cls_data = {"image": cls_img}
                cls_batch = transform(cls_data, self.cls_ops)
                cls_img = cls_batch[0].unsqueeze(dim=0) # image
                cls_img = cls_img.to(self.cls_device)
                cls_preds = self.clser(cls_img)
                cls_post_result = self.cls_post_process_class(cls_preds)
                # parser text and prob
                pred_cls, prob_cls = cls_post_result[0]
                prob_cls = round(prob_cls, 2)
                if pred_cls == "180":
                    # part_img = np.rot90(part_img, 2)
                    part_img = cv2.rotate(part_img, cv2.ROTATE_180)
            # ----- do rec -----#
            if self.rec_img_mode == "GRAY":
                rec_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)
            elif self.rec_img_mode == "RGB":
                rec_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB)
            else:
                rec_img = part_img.copy()
            rec_data = {"image": rec_img}
            rec_batch = transform(rec_data, self.rec_ops)
            rec_img = rec_batch[0].unsqueeze(dim=0) # image
            rec_img = rec_img.to(self.rec_device)
            rec_preds = self.recer(rec_img)
            rec_post_result = self.rec_post_process_class(rec_preds)
            # parser text and prob
            text, prob_rec = rec_post_result[0]
            prob_rec = round(prob_rec, 2)

            ocr_res.append([box, text, prob_rec])
        
        return ocr_res


@torch.no_grad()
def main():
    args = parse_args()
    myOCRer = OCRer(
        args.det_config, args.det_model_path, 
        args.rec_config, args.rec_model_path, 
        args.cls_config, args.cls_model_path, 
        args.character_dict_path,
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
        ocr_res = myOCRer.run(str(img_path))

        # save2txt
        save_txt_path = out_dir.joinpath("res_"+str(img_path.stem)+".txt")
        with open(str(save_txt_path), "w", encoding="UTF-8") as fp:
            for cur_res in ocr_res:
                box, text, prob = cur_res
                tmp = [str(coor) for coor in box.reshape(-1).tolist()]
                tmp.append(text)
                tmp.append(str(prob))
                fp.write(",".join(tmp) + "\n")

        # save2img
        save_img_path = out_dir.joinpath("res_"+str(img_path.stem)+".jpg")
        res_img = draw_ocr_res(ocr_res, str(img_path), str(save_img_path))
        if args.show: 
            cv2.imshow("ocr_res", res_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
