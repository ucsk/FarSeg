import paddle

import paddleseg
from paddleseg import transforms as T
from paddleseg.core import evaluate
from paddleseg.models import FarSeg

eval_transforms = [
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

eval_dataset = paddleseg.datasets.Dataset(
    mode='val',
    num_classes=16,
    dataset_root='../iSAID/valid_dir',
    val_path='../iSAID/valid_dir/val_list.txt',
    transforms=eval_transforms)

model = FarSeg(num_classes=16)
model_state_dict = paddle.load('pretrain_weights/farseg_r50_896x896_asf_amp_60k.pdparams')
model.set_dict(model_state_dict)

mean_iou, acc, class_iou, class_precision, kappa = evaluate(
    model=model,
    eval_dataset=eval_dataset,
    is_slide=True,
    stride=(512, 512),
    crop_size=(896, 896),
    num_workers=4)
