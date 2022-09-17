import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

import pycuda.driver as cuda
import pycuda.autoinit
from trt_utils import get_engine, get_binding_idxs, normlize_cuda

from pytocr.data import create_operators, transform
from pytocr.postprocess import build_post_process
from pytocr.utils.utility import sort_boxes
from utils import load_config, draw_det_res


def parse_args():
    parser = argparse.ArgumentParser(
        description="pytocr det_model infer")
    parser.add_argument("--config", type=str, help="configuration file to use")
    parser.add_argument("--trt_path", type=str, help="model weights file to use")
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
        cfg_path, 
        trt_path,
        gpu_id=0
    ) -> None:
        # ----- detecter init ----- #
        cfg = load_config(cfg_path)
        cfg["Global"]["distributed"] = False
        # check if set use_gpu=True
        device = cuda.Device(gpu_id)
        device_attributes_tuples = device.get_attributes()
        MAX_BLOCK_DIM_X = device_attributes_tuples[pycuda._driver.device_attribute.MAX_BLOCK_DIM_X]
        # load trt engine
        engine = get_engine(trt_path)    
        context = engine.create_execution_context()
        context.active_optimization_profile = 0
        input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, context.active_optimization_profile)
        # build post process
        post_process_class = build_post_process(cfg["PostProcess"], cfg["Global"])
        # create data ops
        transforms = []
        img_mode = "RGB"
        norm_mean = [0.0, 0.0, 0.0]
        norm_std = [1.0, 1.0, 1.0]
        for op in cfg["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "DecodeImage" in op_name:
                img_mode = op[op_name]["img_mode"]
                continue
            elif ("Label" in op_name) or (
                "ToTensor" in op_name):
                continue
            elif "Normalize" in op_name:
                norm_mean = op[op_name]["mean"]
                norm_std = op[op_name]["std"]
                continue
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image", "shape"]
            transforms.append(op)
        ops = create_operators(transforms, cfg["Global"])
        self.engine = engine
        self.context = context
        self.MAX_BLOCK_DIM_X = MAX_BLOCK_DIM_X
        self.input_binding_idxs = input_binding_idxs
        self.output_binding_idxs = output_binding_idxs
        self.post_process_class = post_process_class
        self.ops = ops
        self.img_mode = img_mode
        self.normlize = normlize_cuda()  # output [c h w]
        self.norm_mean = np.array(norm_mean, dtype=np.float32)
        self.norm_std = np.array(norm_std, dtype=np.float32)
        
    def get_engine_inputs(self, img):
        h, w, c = img.shape[:3]
        block_x = self.MAX_BLOCK_DIM_X
        grid_x = (h * w + block_x - 1) // block_x
        device_mem = cuda.mem_alloc(h * w * c * 4) # float32 4字节 nbytes
        mean_gpu = cuda.mem_alloc(self.norm_mean.nbytes)
        cuda.memcpy_htod(mean_gpu, self.norm_mean)
        std_gpu = cuda.mem_alloc(self.norm_std.nbytes)
        cuda.memcpy_htod(std_gpu, self.norm_std)
        self.normlize(
            cuda.In(img), device_mem, 
            np.uint32(h), np.uint32(w), 
            mean_gpu, std_gpu, 
            block=(block_x, 1, 1), 
            grid=(grid_x, 1, 1))  # 只获取device结果，并返回
        device_inputs = [device_mem]
        return device_inputs
    
    def get_engine_outputs(self):
        host_outputs = []
        device_outputs = []
        for binding_index in self.output_binding_idxs:
            output_shape = self.context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(output_shape, dtype=np.float32)
            host_outputs.append(buffer)
            # Allocate output buffers on device
            device_outputs.append(cuda.mem_alloc(buffer.nbytes))  
        return device_outputs, host_outputs
    
    def run(self, img_path):
        ori_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.img_mode == "RGB":
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        else:
            img = ori_img.copy()
        data = {"image": img}
        batch = transform(data, self.ops)
        img = batch[0] # image
        h, w, c = img.shape[:3]
        shape_list = np.expand_dims(batch[1], axis=0)   # add batch dim
        # do trt infer
        device_inputs = self.get_engine_inputs(img)
        self.context.set_binding_shape(0, (1, c, h, w)) # dynamic input size     
        device_outputs, host_outputs = self.get_engine_outputs()
        bindings = device_inputs + device_outputs   # merge
        self.context.execute_v2(bindings)  
        cuda.memcpy_dtoh(host_outputs[-1], device_outputs[-1])
        preds = {}
        preds["maps"] = host_outputs[-1]
        post_result = self.post_process_class(preds, shape_list)
        # parser boxes if post_result is dict
        boxes = sort_boxes(post_result[0]["points"])  # 由上到下 由左到右 排版
        
        return boxes


def main():
    args = parse_args()
    myDeter = Deter(args.config, args.trt_path, args.gpu_id)

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
        det_res = myDeter.run(str(img_path))

        # save2txt
        save_txt_path = out_dir.joinpath("res_"+str(img_path.stem)+".txt")
        with open(str(save_txt_path), "w", encoding="UTF-8") as fp:
            for box in det_res:
                box = [str(coor) for coor in box.reshape(-1).tolist()]
                fp.write(",".join(box) + "\n")
        
        # save2img
        save_img_path = out_dir.joinpath("res_"+str(img_path.stem)+".jpg")
        res_img = draw_det_res(det_res, str(img_path), str(save_img_path))
        if args.show: 
            cv2.imshow("det_res", res_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
