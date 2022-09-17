# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is based on https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md
# The copyright of Open-MMLab is as follows:
# Apache License (see https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE for details).

import argparse
import glob
import numbers
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from tqdm import tqdm

iSAID_palette = \
    {
        0: (0, 0, 0),
        1: (0, 0, 63),
        2: (0, 63, 63),
        3: (0, 63, 0),
        4: (0, 63, 127),
        5: (0, 63, 191),
        6: (0, 63, 255),
        7: (0, 127, 63),
        8: (0, 127, 127),
        9: (0, 0, 127),
        10: (0, 0, 191),
        11: (0, 0, 255),
        12: (0, 191, 127),
        13: (0, 127, 191),
        14: (0, 127, 255),
        15: (0, 100, 155)
    }

iSAID_invert_palette = {v: k for k, v in iSAID_palette.items()}


def iSAID_convert_from_color(arr_3d, palette=iSAID_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def impad(img, shape=None, pad_val=0):
    width = max(shape[1] - img.shape[1], 0)
    height = max(shape[0] - img.shape[0], 0)
    padding = (0, 0, width, height)

    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=pad_val)

    return img


def slide_crop_image(src_path, out_dir, mode, patch_H, patch_W, overlap):
    img = cv2.imread(src_path)
    img_H, img_W, _ = img.shape

    if img_H < patch_H and img_W > patch_W:
        img = impad(img, shape=(patch_H, img_W), pad_val=0)
        img_H, img_W, _ = img.shape

    elif img_H > patch_H and img_W < patch_W:
        img = impad(img, shape=(img_H, patch_W), pad_val=0)
        img_H, img_W, _ = img.shape

    elif img_H < patch_H and img_W < patch_W:
        img = impad(img, shape=(patch_H, patch_W), pad_val=0)
        img_H, img_W, _ = img.shape

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            image = osp.basename(src_path).split('.')[0] + '_' + str(
                y_str) + '_' + str(y_end) + '_' + str(x_str) + '_' + str(
                x_end) + '.png'
            save_path = osp.join(out_dir, 'img_dir', mode, str(image))
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, img[y_str:y_end, x_str:x_end, :])


def slide_crop_label(src_path, out_dir, mode, patch_H, patch_W, overlap):
    label = cv2.imread(src_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    label = iSAID_convert_from_color(label)
    img_H, img_W = label.shape

    if img_H < patch_H and img_W > patch_W:
        label = impad(label, shape=(patch_H, img_W), pad_val=255)
        img_H = patch_H

    elif img_H > patch_H and img_W < patch_W:
        label = impad(label, shape=(img_H, patch_W), pad_val=255)
        img_W = patch_W

    elif img_H < patch_H and img_W < patch_W:
        label = impad(label, shape=(patch_H, patch_W), pad_val=255)
        img_H = patch_H
        img_W = patch_W

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            image = osp.basename(src_path).split('.')[0].split(
                '_')[0] + '_' + str(y_str) + '_' + str(y_end) + '_' + str(
                x_str) + '_' + str(x_end) + '_instance_color_RGB' + '.png'
            save_path = osp.join(out_dir, 'ann_dir', mode, str(image))
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, label[y_str:y_end, x_str:x_end])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='iSAID folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')

    parser.add_argument(
        '--patch_width',
        default=896,
        type=int,
        help='Width of the cropped image patch')
    parser.add_argument(
        '--patch_height',
        default=896,
        type=int,
        help='Height of the cropped image patch')
    parser.add_argument(
        '--overlap_area', default=384, type=int, help='Overlap area')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    patch_H, patch_W = args.patch_width, args.patch_height
    overlap = args.overlap_area

    if args.out_dir is None:
        out_dir = osp.join('iSAID')
    else:
        out_dir = args.out_dir

    os.makedirs(osp.join(out_dir, 'img_dir', 'train'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'img_dir', 'val'), exist_ok=True)

    os.makedirs(osp.join(out_dir, 'ann_dir', 'train'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'ann_dir', 'val'), exist_ok=True)

    os.makedirs(osp.join(out_dir, 'valid_dir', 'images'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'valid_dir', 'labels'), exist_ok=True)

    assert os.path.exists(os.path.join(dataset_path, 'train')), \
        'train is not in {}'.format(dataset_path)
    assert os.path.exists(os.path.join(dataset_path, 'val')), \
        'val is not in {}'.format(dataset_path)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for dataset_mode in ['train', 'val']:
            img_zipp_list = glob.glob(
                os.path.join(dataset_path, dataset_mode, 'images', '*.zip'))
            for img_zipp in img_zipp_list:
                zip_file = zipfile.ZipFile(img_zipp)
                zip_file.extractall(os.path.join(tmp_dir, dataset_mode, 'img'))

            label_zipp_list = glob.glob(
                os.path.join(dataset_path, dataset_mode, 'Semantic_masks', '*.zip'))
            for label_zipp in label_zipp_list:
                zip_file = zipfile.ZipFile(label_zipp)
                zip_file.extractall(os.path.join(tmp_dir, dataset_mode, 'lab'))

            src_path_list = glob.glob(
                os.path.join(tmp_dir, dataset_mode, 'img', 'images', '*.png'))
            for img_path in tqdm(src_path_list, desc=f'{out_dir}/img_dir/{dataset_mode}', ncols=100):
                slide_crop_image(img_path, out_dir, dataset_mode,
                                 patch_H, patch_W, overlap)

            lab_path_list = glob.glob(
                os.path.join(tmp_dir, dataset_mode, 'lab', 'images', '*.png'))
            for lab_path in tqdm(lab_path_list, desc=f'{out_dir}/ann_dir/{dataset_mode}', ncols=100):
                slide_crop_label(lab_path, out_dir, dataset_mode,
                                 patch_H, patch_W, overlap)

            with open(os.path.join(out_dir, f'{dataset_mode}_list.txt'), 'w') as list_file:
                image_list = sorted(os.listdir(os.path.join(out_dir, f'img_dir/{dataset_mode}')))
                label_list = sorted(os.listdir(os.path.join(out_dir, f'ann_dir/{dataset_mode}')))
                assert len(image_list) == len(label_list)
                for i in range(len(image_list)):
                    img_name, mask_name = image_list[i], label_list[i]
                    assert img_name.split('.')[0] == mask_name.split('_inst')[0]
                    list_file.write(f'img_dir/{dataset_mode}/{img_name} ann_dir/{dataset_mode}/{mask_name}\n')

            if dataset_mode == 'val':
                for src_path in tqdm(src_path_list, desc=f'{out_dir}/valid_dir/images', ncols=100):
                    image_name = os.path.normpath(src_path).split(os.sep)[-1]
                    shutil.copyfile(src_path, os.path.join(out_dir, 'valid_dir', 'images', image_name))
                for lab_path in tqdm(lab_path_list, desc=f'{out_dir}/valid_dir/labels', ncols=100):
                    label_name = os.path.normpath(lab_path).split(os.sep)[-1]
                    label = cv2.imread(lab_path)
                    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                    label = iSAID_convert_from_color(label)
                    cv2.imwrite(os.path.join(out_dir, 'valid_dir', 'labels', label_name), label)

                with open(os.path.join(out_dir, 'valid_dir', f'val_list.txt'), 'w') as list_file:
                    image_list = sorted(os.listdir(os.path.join(out_dir, 'valid_dir/images')))
                    label_list = sorted(os.listdir(os.path.join(out_dir, 'valid_dir/labels')))
                    assert len(image_list) == len(label_list)
                    for i in range(len(image_list)):
                        image_name, label_name = image_list[i], label_list[i]
                        assert image_name.split('.')[0] == label_name.split('_inst')[0]
                        list_file.write(f'images/{image_name} labels/{label_name}\n')

        with open(os.path.join(out_dir, 'labels.txt'), 'w') as label_file:
            label_file.write('background\nShip\nST\nBD\nTC\nBC\nGTF\n'
                             'Bridge\nLV\nSV\nHC\nSP\nRA\nSBF\nPlane\nHarbor\n')


if __name__ == '__main__':
    main()
