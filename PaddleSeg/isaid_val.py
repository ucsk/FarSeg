import glob
import os.path

import paddle
from visualdl import LogWriter

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
    dataset_root='../iSAID',
    val_path='../iSAID/val_list.txt',
    transforms=eval_transforms)

log_writer = LogWriter(logdir='vdl_logs')
params_dir_list = glob.glob('farseg_r50_896x896_asf_amp_60k/iter_*')
params_dir_list = sorted(params_dir_list, key=os.path.getctime)
for params_dir in params_dir_list:
    params_path = os.path.join(params_dir, 'model.pdparams')
    print(params_path)
    if not os.path.exists(params_path):
        continue

    model = FarSeg(num_classes=16)
    model_state_dict = paddle.load(params_path)
    model.set_dict(model_state_dict)

    mean_iou, acc, class_iou, class_precision, kappa = evaluate(
        model=model,
        eval_dataset=eval_dataset,
        num_workers=4)

    iter_num = int(params_dir.split('iter_')[-1])
    log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter_num)
    log_writer.add_scalar('Evaluate/Acc', acc, iter_num)
    log_writer.add_scalar('Evaluate/kappa', kappa, iter_num)

    print()

log_writer.close()
