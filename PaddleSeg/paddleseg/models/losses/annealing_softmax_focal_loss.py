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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class AnnealingSoftmaxFocalLoss(nn.Layer):
    def __init__(self, T_max=10000, gamma=2.0, ignore_index=255):
        super().__init__()
        self.T = 0
        self.T_max = T_max
        self.gamma = gamma
        self.ignore_index = ignore_index

    def cosine_annealing(self, lower_bound, upper_bound):
        return upper_bound + 0.5 * (lower_bound - upper_bound) * \
               (math.cos(math.pi * self.T / self.T_max) + 1)

    def forward(self, logit, label):
        ce_loss = F.cross_entropy(
            paddle.transpose(logit, [0, 2, 3, 1]), label,
            ignore_index=self.ignore_index, reduction='none')

        with paddle.no_grad():
            p = F.softmax(logit, axis=1)
            valid_mask = ~ label.equal(self.ignore_index)
            masked_y_true = paddle.where(valid_mask, label, paddle.zeros_like(label))
            index = masked_y_true.unsqueeze(axis=1)
            modulating_factor = (1 - p).pow(self.gamma)
            modulating_factor = paddle.take_along_axis(modulating_factor, indices=index, axis=1)
            modulating_factor = modulating_factor.squeeze_(axis=1)
            normalizer = paddle.sum(ce_loss) / paddle.sum(ce_loss * modulating_factor)
            scales = modulating_factor * normalizer

        if self.training and self.T < self.T_max:
            self.T += 1
            scale = self.cosine_annealing(1, scales)
        else:
            scale = scales

        asf_loss = paddle.sum(ce_loss * scale) / (paddle.sum(valid_mask) + p.shape[0])

        return asf_loss
