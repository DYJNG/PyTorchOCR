# PyTorchOCR
## 简介
PyTorchOCR是一套基于PyTorch的实用OCR工具库。

***
## 写在前面
- 项目以PaddleOCR为模板，结合相关OCR算法的PyTorch实现库，对核心OCR检测、识别等算法进行提取、整理与微调，易于阅读与扩展。
- 决定开发该项目的目的主要是为了梳理OCR整个框架和学习相关算法的实现细节。
- 项目初步计划实现以下几方面内容：
    - OCR检测
        - [x] DBNet & DBNet++ (速度优先)
        - [x] PSENet (精度优先)
        - [x] PANet & PANet++ (trade-off)
    - OCR识别
        - [x] CRNN
        - [x] STAR-Net（TPS-场景文本识别）
    - 方向分类
        - [x] 文本行方向分类
    - 关键字段提取
    - 模型轻量化
        - [ ] 剪枝
        - [x] 蒸馏
        - [ ] 量化
        - [x] 后处理加速
    - 模型部署
        - [x] onnx
        - [x] tensorrt
        - [ ] ncnn
    - OCR应用
        - [ ] 身份证识别
        - [ ] 表格识别

***
## 近期更新
- 2022.10.01 支持OCR识别STAR-Net，新增TPS识别transforms。
- 2022.08.26 新增onnx、tensorrt转换脚本及推理代码。
- 2022.08.22 新增pytorch推理代码。
- 2022.06.13 新增det_swin、det_convnext检测backbone。
- 2022.05.09 支持OCR检测DBNet++。
- 2022.05.06 支持OCR检测PANet和PANet++。
- 2022.04.28 支持OCR检测PSENet。
- 2022.04.26 新增det_repvgg检测backbone。
- 2022.04.25 支持OCR文本行分类。
- 2022.04.24 支持OCR识别CRNN，新增rec_vgg、rec_mobilenet_v3、rec_resnet等识别backbone。
- 2022.04.20 支持知识蒸馏distill、DML、CML。
- 2022.04.18 新增det_mobilenet_v3、det_shufflenet_v2检测backbone。
- 2022.04.15 支持OCR检测DBNet，新增det_resnet检测backbone，支持后处理CPP加速。
- 2022.04.13 整体目录结构搭建完毕，支持分布式训练，Apex混合精度(Apex分布式方式训练尚未验证)。

*** 
## 模型效果
- ### 文本检测（测试时输入尺寸统一设置为：短边固定736，长边等比例resize。若增大输入尺寸，则指标会有一定提升。）
    - ICDAR-2015
        |模型|骨干网络|大小|precision|recall|hmean|
        |:-:|:-:|:-:|:-:|:-:|:-:|
        |DB|r50|98M|86.93|73.95|79.92|
        |DB|r18|48M|85.57|73.38|79.00|
        |DB|mbv3smallx1.0|4.5M|80.34|56.86|66.5|
        |DB|mbv3smallx1.0_distill|4.5M|82.88|52.67|64.41|
        |DB|mbv3smallx1.0_dml|4.5M|79.89|58.35|67.45|
        |DB|mbv3smallx1.0_cml|4.5M|82.16|59.41|68.96|
        |DB++|r18|48M|86.00|73.62|79.33|
        |PSE|r50|112M|79.45|75.20|77.27|
        |PA++|r18|47M|82.79|75.25|78.84|
***
- ### 文本识别（TODO）
    - ICDAR-2015
        |模型|骨干网络|大小|score|map|
        |-|-|-|-|-|
***
- ### 模型下载（提取码：t5w4）
    - OCR检测（训练数据主要来自天池“MTWI-2018”+“英特尔创新大师杯”OCR竞赛）
        |模型|骨干网络|大小|model|
        |:-:|:-:|:-:|:-:|
        |DB|r50|98M|[ckpts/det/torch/db_r18.pth](https://pan.baidu.com/s/1IQu2vv5sS8lHdvXPRtgvsA)|
        |DB|r18|48M|[ckpts/det/torch/db_r50.pth](https://pan.baidu.com/s/1IQu2vv5sS8lHdvXPRtgvsA)|
        |DB|mbv3large_x1.0_cml|13M|[ckpts/det/torch/db_mbv3large_x1.0_cml.pth](https://pan.baidu.com/s/1IQu2vv5sS8lHdvXPRtgvsA)|
        |DB++|r18|48M|[ckpts/det/torch/dbplusplus_r18.pth](https://pan.baidu.com/s/1IQu2vv5sS8lHdvXPRtgvsA)|
        |PSE|r50|112M|[ckpts/det/torch/pse_r50.pth](https://pan.baidu.com/s/1IQu2vv5sS8lHdvXPRtgvsA)|
        |PA++|r18|47M|[ckpts/det/torch/paplusplus_r18.pth](https://pan.baidu.com/s/1IQu2vv5sS8lHdvXPRtgvsA)|
    - OCR识别（训练数据主要来自生成（200W+文本行）+天池“英特尔创新大师杯”OCR竞赛）
        |模型|骨干网络|大小|model|
        |:-:|:-:|:-:|:-:|
        |CRNN|vgg_v1_x1.0_gray|45M|[ckpts/rec/crnn_vgg_v1_x1.0_gray_6623.pth](https://pan.baidu.com/s/1IQu2vv5sS8lHdvXPRtgvsA)|
***
- ### 结果展示
<img src="figs/wlms.png" width="730px"/> 

***
## 快速开始
- ### 环境配置
    ```
    cmake-3.18.4.post1
    gcc-5.4.0
    opencv-3.4.2
    # 以上为OCR检测后处理加速C++代码编译需要，若无需加速，可在config文件设置[PostProcess.cpp_speedup: False]
    ```
    ```
    pip install -r requirement.txt
    ```
- ### 数据集整理
    ```
    # 检测数据集
    python tools/data/gen_json_label.py \
        --mode det \
        --img_dir .../OCR_DET_DATASETS/images \ 
        --label_dir .../OCR_DET_DATASETS/labels \            # *.txt -> x1, y1, x2, y2, x3, y3, x4, y4  和图片一一对应
        --out_path .../OCR_DET_DATASETS/train_label.txt \    # 训练集标签保存路径
        --sort_pts True \                                    # 检测框坐标排序
        --split_train_val True --ratio 0.95 \                # 按照设定比例随机划分训练集和验证集
        --extra_out_path .../OCR_DET_DATASETS/val_label.txt  # 验证集标签保存路径
    ```
    ```
    # 识别数据集
    python tools/data/gen_json_label.py \
        --mode rec \
        --img_dir .../OCR_REC_DATASETS/images \               # 文本行图片
        --label_dir .../OCR_REC_DATASETS/labels \             # *.txt -> text_contents 文本内容
        --out_path .../OCR_REC_DATASETS/train_label.txt \
        --split_train_val True --ratio 0.97 \
        --extra_out_path .../OCR_REC_DATASETS/val_label.txt
    ```
- ### 模型训练
    ```
    # 单卡
    CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/det/det_r18_db.yml -o Global.distributed=False
    ```
    ```
    # 多卡
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tools/train.py -c configs/det/det_r18_db.yml
    ```
- ### 模型推理
    #### [pytorch](deploy/pytorch/)
    ```
    # 检测
    python deploy/pytorch/infer_det.py \
        --config configs/det/det_r18_db.yml \
        --model_path models/torch/db_r18.pth \
        --img_path path_to_test_img/dir \
        --out_dir path_to_out_dir

    # 识别
    python deploy/pytorch/infer_rec.py \
        --config configs/rec/rec_vgg_bilstm_ctc.yml \
        --model_path path_to_rec_model \
        --character_dict_path path_to_character_dict \
        --img_path path_to_textline_img/dir \
        --out_dir path_to_out_dir

    # 检测+识别
    python deploy/pytorch/run_ocr.py \
        --det_config path_to_det_config \
        --det_model_path path_to_det_model \
        --rec_config path_to_rec_config \
        --rec_model_path path_to_rec_model \
        --img_path path_to_test_img/dir \
        --out_dir path_to_out_dir
    ```
    #### [tensorrt](deploy/tensorrt/)
    ```
    # 检测
    python deploy/pytorch/infer_det.py \
        --config configs/det/det_r18_db.yml \
        --trt_path path_to_trt_engine \
        --img_path path_to_test_img/dir \
        --out_dir path_to_out_dir
    ```

***
## 参考及引用
- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/WenmuZhou/PytorchOCR
- https://github.com/BADBADBADBOY/pytorchOCR
- https://github.com/WenmuZhou/DBNet.pytorch
- https://github.com/whai362/PSENet
- https://github.com/whai362/pan_pp.pytorch