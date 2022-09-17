# FarSeg (Implemented by PaddlePaddle 2.3)

## 1. 简介

论文针对高分辨率遥感影像分割场景中的典型问题（尺度变化，背景类内方差不均衡、前景-背景不均衡）提出FarSeg。FarSeg在网络结构上采用FPN处理多尺度问题，设计了前景-场景关系模块用于加强前景类间的联系、增大前景类-背景类差异，然后提出前景优化方法缓解类别不均衡问题。在iSAID验证集上，FarSeg和自然场景通用分割模型相比，指标验证了方法的有效性。

![](https://raw.githubusercontent.com/Z-Zheng/images_repo/master/farseg.png)

**论文：** [Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery](https://arxiv.org/abs/2011.09766)

**参考repo：** [Zheng: FarSeg](https://github.com/Z-Zheng/FarSeg)

在此非常感谢 [Zheng](https://github.com/Z-Zheng) 等人贡献的FarSeg项目，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

- 航空图像数据集iSAID—语义分割部分：[https://captain-whu.github.io/iSAID/index.html](https://captain-whu.github.io/iSAID/index.html)

复现精度如下。

| method   | iters | bs   | card | loss    | align_corners | mIoU  | weight                                                       | log                                                          |
| -------- | ----- | ---- | ---- | ------- | ------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| official | 60k   | 4    | 2    | ASFLoss | √             | 63.71 | [download](https://github.com/Z-Zheng/FarSeg/releases/download/v1.0/farseg50.pth) | \                                                            |
| ours     | 60k   | 8    | 1    | ASFLoss | √             | 63.09 | [download](https://bj.bcebos.com/v1/ai-studio-online/0e0057eb768d42d7b8f3389b0114cf70f74cdf65c0a749eb8f135b37ee06306a?responseContentDisposition=attachment%3B%20filename%3Dfarseg_r50_896x896_asf_amp_60k.pdparams&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2022-09-15T08%3A01%3A37Z%2F-1%2F%2Fe663278bfc9afc49a0542e8c212ff11d06c276675050554b14d2ecd15e252add) | [log_dir](https://github.com/ucsk/FarSeg/tree/main/PaddleSeg/vdl_logs/) 丨[train](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=e8d4f8450274b9a0f99e89eee7d8cb71) & [val](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=866f4a62e9c52bc083c09c8fd67c6e0b) |

关于模型验证指标，首先使用常规方法在11644张验证集上进行，然后将前一步中的最优模型（mIoU=62.56）在原始的458张验证集图像上执行滑窗预测，计算得到最终结果（mIoU=63.09）。


## 3. 准备数据与环境

### 3.1 准备环境

- 环境：AI Studio & BML CodeLab & Python 3.7；
- 硬件：Nvidia Tesla V100 32G × 1；
- 框架：PaddlePaddle 2.3.2；

```jupyter
!git clone https://github.com/ucsk/FarSeg.git
%cd FarSeg/
```

### 3.2 准备数据

训练和评估所需数据为iSAID数据集的语义分割任务部分。

图像可以在 <a href="https://captain-whu.github.io/DOTA/dataset.html" target="_blank">DOTA-v1.0</a> (train/val/test)下载，标注可在 <a href="https://captain-whu.github.io/iSAID/dataset.html" target="_blank">iSAID</a> (train/val)下载。

对于下载完成的原始iSAID数据集，按照如下结构在根目录进行准备进行准备：

```diff
├── PaddleRS
├── PaddleSeg
+└── iSAID_zip
+    ├── train
+    │   ├── images
+    │   │   ├── part1.zip
+    │   │   ├── part2.zip
+    │   │   └── part3.zip
+    │   └── Semantic_masks
+    │       └── images.zip
+    └── val
+        ├── images
+        │   └── part1.zip
+        └── Semantic_masks
+            └── images.zip
```

准备好原始数据集的目录结构之后，在项目根目录按如下命令生成步长为512、尺寸为896x896的图像（若尺寸不足，图像填充0，标签填充255）。

```jupyter
!python PaddleRS/tools/prepare_dataset/prepare_isaid.py iSAID_zip
```

经过预处理后的数据会存储在根目录`iSAID/`中，其描述如下，其中`valid_dir/`为验证集原图，`labels.txt`包含背景类：

- [AI Studio: iSAID (patch)](https://aistudio.baidu.com/aistudio/datasetdetail/167913) ；
- 概况：16类（含背景），训练集33978张，验证集11644张；
- 数据格式：图片为RGB三通道图像，标签为单通道图像，值为INT[0,15]+{255}，二者均为PNG格式存储；

```diff
PaddleRS
PaddleSeg
iSAID_zip
+iSAID
+├── ann_dir
+│   ├── train
+│   └── val
+├── img_dir
+│   ├── train
+│   └── val
+├── valid_dir
+│   ├── images
+│   ├── labels
+│   └── val_list.txt
+├── train_list.txt
+├── val_list.txt
+└── labels.txt
```


## 4. 开始使用

考虑到PaddleRS使用epoch作为checkpoint保存单位，所以在训练和评估阶段使用PaddleSeg接口实现。

Jupyter环境工作目录切换到`PaddleSeg/`。

```jupyter
%cd PaddleSeg/
!pip install -r requirements.txt
!python setup.py install
```

### 4.1 模型训练

主要训练配置如下：

-   模型训练步长60000，单卡训练批大小设为8；
-   优化器：SGD（momentum=0.9, weight_decay=1e-4, [clip_grad_by_norm](https://github.com/ucsk/FarSeg/blob/main/PaddleSeg/paddleseg/cvlibs/config.py#L228) ）；
-   学习率策略：多项式衰减PolynomialDecay（begin=0.007，end=0.0, power=0.9）；

在终端环境中下载由X2Paddle转换得到的ResNet50预训练权重，保存在`pretrain_weights/resnet50_pth.pdparams`。

```bash
cd FarSeg/PaddleSeg/
wget -O pretrain_weights/resnet50_pth.pdparams https://bj.bcebos.com/v1/ai-studio-online/ffef16b1a0004f1bba606c7540d501ea5cf116a2288742b4888cf88c64ee0f55?responseContentDisposition=attachment%3B%20filename%3Dresnet50_pth.pdparams&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2022-09-11T06%3A46%3A52Z%2F-1%2F%2F1bdd0efebc7f95167ab8bf22a434c6c449647bfa57fa814ab588e03dc8a76bbc
```

开始训练，单GPU自动混合精度。

```jupyter
!python train.py \
    --config=configs/farseg/farseg_r50_896x896_asf_amp_60k.yml \
    --save_interval=400 \
    --keep_checkpoint_max=50 \
    --num_workers=4 \
    --log_iters=50 \
    --save_dir=farseg_r50_896x896_asf_amp_60k \
    --use_vdl \
    --precision=fp16 \
    --amp_level=O1
```

- 训练日志可视化：<a href="https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=e8d4f8450274b9a0f99e89eee7d8cb71" target="_blank">VisualDL - train</a>

### 4.2 模型评估

训练时未进行评估，待训练完成后单独对已保存模型进行评估，并写入日志，选择验证集最优模型。

```jupyter
!python isaid_val.py
```

- 评估日志可视化：<a href="https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=866f4a62e9c52bc083c09c8fd67c6e0b" target="_blank">VisualDL - valid</a>

以下内容中，单独对本repo训练好的最优权重进行评估。

FarSeg权重文件和ResNet50预训练权重一样，在终端`FarSeg/PaddleSeg/`路径中进行下载。

```bash
cd FarSeg/PaddleSeg/
wget -O pretrain_weights/farseg_r50_896x896_asf_amp_60k.pdparams https://bj.bcebos.com/v1/ai-studio-online/0e0057eb768d42d7b8f3389b0114cf70f74cdf65c0a749eb8f135b37ee06306a?responseContentDisposition=attachment%3B%20filename%3Dfarseg_r50_896x896_asf_amp_60k.pdparams&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2022-09-15T08%3A01%3A37Z%2F-1%2F%2Fe663278bfc9afc49a0542e8c212ff11d06c276675050554b14d2ecd15e252add
```

- 已切图的验证集上进行评估：

```jupyter
!python val.py \
    --config=configs/farseg/farseg_r50_896x896_asf_amp_60k.yml \
    --model_path=pretrain_weights/farseg_r50_896x896_asf_amp_60k.pdparams \
    --num_workers=4
```

- 验证集原图上进行滑窗评估：

```jupyter
!python val.py \
    --config=configs/farseg/farseg_r50_slide_val.yml \
    --model_path=pretrain_weights/farseg_r50_896x896_asf_amp_60k.pdparams \
    --num_workers=4 \
    --is_slide \
    --crop_size 896 896 \
    --stride 512 512
```


## 5. 模型推理部署

这里手动生成了`PaddleRS/normal_model/model.yml`，其中包含了使PaddleRS成功调用模型的参数。

FarSeg模型迁移在这里 [paddlers/rs_models/seg](https://github.com/ucsk/FarSeg/tree/main/PaddleRS/paddlers/rs_models/seg) 。

将工作目录切换到`FarSeg/PaddleRS`。然后安装依赖，拷贝[4.2]小节下载的权重，导出部署模型。

```jupyter
%cd ../PaddleRS/
!pip install -r requirements.txt
!python setup.py install
```

```jupyter
!cp ../PaddleSeg/pretrain_weights/farseg_r50_896x896_asf_amp_60k.pdparams normal_model/model.pdparams
!python deploy/export/export_model.py --model_dir=normal_model --save_dir=inference_model
```

运行动态图、静态图的加载与预测，可视化图像保存在`infer_test/`。

```jupyter
!python infer_test/infer.py
```


## 6. TIPC测试

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的流程是否能走通，不验证精度和速度；

```jupyter
!pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
!bash ./test_tipc/prepare.sh test_tipc/configs/seg/farseg/train_infer_python.txt lite_train_lite_infer
```

AI Studio环境中 [TIPC分布式训练](https://github.com/ucsk/FarSeg/blob/main/PaddleRS/test_tipc/results.log#L6) 有时会报错“Termination signal”。

本repo将运行环境切换为V100×4后成功运行，单卡有时也可以。

```jupyter
!bash ./test_tipc/test_train_inference_python.sh test_tipc/configs/seg/farseg/train_infer_python.txt lite_train_lite_infer
```

TIPC测试日志文件保存于 [test_tipc/output/results.log](https://github.com/ucsk/FarSeg/blob/main/PaddleRS/test_tipc/results.log) 。

## 7. LICENSE

本项目的发布受 [Apache 2.0 license](https://github.com/ucsk/FarSeg/blob/develop/LICENSE) 许可认证。

## 8. 参考链接与文献

- 论文： <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.pdf" target="_blank">Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery</a>
- 代码： <a href="https://github.com/Z-Zheng/FarSeg" target="_blank">FarSeg (Zheng)</a> 、 <a href="https://github.com/PaddlePaddle/PaddleRS" target="_blank">FarSeg (PaddleRS)</a>
