# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager
from paddleseg.models.backbones import ResNet50Encoder
from paddleseg.models.layers import initializer as init


class FPNConvBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1):
        super(FPNConvBlock, self).__init__()

        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias_attr=True)
        init.kaiming_uniform_(self._conv.weight, a=1)
        init.constant_(self._conv.bias, 0)

    def forward(self, x):
        x = self._conv(x)
        return x


class DefaultConvBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 weight_attr=None,
                 bias_attr=None):
        super(DefaultConvBlock, self).__init__()

        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        init.kaiming_uniform_(self._conv.weight, a=math.sqrt(5))
        if self._conv.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self._conv.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self._conv.bias, -bound, bound)

    def forward(self, x):
        x = self._conv(x)
        return x


class LastLevelMaxPool(nn.Layer):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()

        self.p6 = nn.Conv2D(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2D(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            for m in module.sublayers():
                init.kaiming_uniform_(m.weight, a=1)
                init.constant_(m.bias, value=0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class FPN(nn.Layer):
    def __init__(self,
                 in_channel_list,
                 out_channels,
                 conv_block=FPNConvBlock,
                 top_blocks=None):
        super(FPN, self).__init__()

        inner_blocks = []
        layer_blocks = []
        for idx, in_channels in enumerate(in_channel_list, 1):
            if in_channels == 0:
                continue
            inner_blocks.append(conv_block(in_channels, out_channels, 1))
            layer_blocks.append(conv_block(out_channels, out_channels, 3, 1))

        self.inner_blocks = nn.LayerList(inner_blocks)
        self.layer_blocks = nn.LayerList(layer_blocks)
        self.top_blocks = top_blocks

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for i, feature in enumerate(x[-2::-1]):
            inner_block = self.inner_blocks[len(self.inner_blocks) - 2 - i]
            layer_block = self.layer_blocks[len(self.layer_blocks) - 2 - i]
            inner_top_down = F.interpolate(
                last_inner, scale_factor=2, mode="nearest")
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class SceneRelation(nn.Layer):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True,
                 conv_block=DefaultConvBlock):
        super(SceneRelation, self).__init__()

        self.scale_aware_proj = scale_aware_proj
        if scale_aware_proj:
            self.scene_encoder = nn.LayerList([
                nn.Sequential(
                    conv_block(in_channels, out_channels, 1),
                    nn.ReLU(),
                    conv_block(out_channels, out_channels, 1))
                for _ in range(len(channel_list))
            ])
        else:
            self.scene_encoder = nn.Sequential(
                conv_block(in_channels, out_channels, 1),
                nn.ReLU(),
                conv_block(out_channels, out_channels, 1))
        self.content_encoders = nn.LayerList()
        self.feature_reencoders = nn.LayerList()
        for channel in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    conv_block(channel, out_channels, 1, bias_attr=True),
                    nn.BatchNorm2D(out_channels, momentum=0.1),
                    nn.ReLU()))
            self.feature_reencoders.append(
                nn.Sequential(
                    conv_block(channel, out_channels, 1, bias_attr=True),
                    nn.BatchNorm2D(out_channels, momentum=0.1),
                    nn.ReLU()))
        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        content_feats = [
            c_en(p_feat)
            for c_en, p_feat in zip(self.content_encoders, features)
        ]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [
                self.normalizer((sf * cf).sum(axis=1, keepdim=True))
                for sf, cf in zip(scene_feats, content_feats)
            ]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [
                self.normalizer((scene_feat * cf).sum(axis=1, keepdim=True))
                for cf in content_feats
            ]
        p_feats = [
            op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)
        ]
        refined_feats = [r * p for r, p in zip(relations, p_feats)]
        return refined_feats


class FarSegNeck(nn.Layer):
    def __init__(self,
                 fpn_channel_list=(256, 512, 1024, 2048),
                 fpn_out_channels=256,
                 sr_in_channels=2048,
                 sr_channel_list=(256, 256, 256, 256),
                 sr_out_channels=256,
                 scale_aware_proj=True):
        super(FarSegNeck, self).__init__()

        self.fpn = FPN(fpn_channel_list, fpn_out_channels)
        self.global_avg_pool2d = nn.AdaptiveAvgPool2D(1)
        self.sr = SceneRelation(sr_in_channels,
                                sr_channel_list,
                                sr_out_channels,
                                scale_aware_proj)

    def forward(self, x):
        features = self.fpn(x)
        scene_feature = self.global_avg_pool2d(x[-1])
        refined_features = self.sr(scene_feature, features)
        return refined_features


class AsymmetricDecoder(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 conv_block=DefaultConvBlock):
        super(AsymmetricDecoder, self).__init__()

        self.blocks = nn.LayerList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feat_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                nn.Sequential(*[
                    nn.Sequential(
                        conv_block(
                            in_channels if idx == 0 else out_channels,
                            out_channels,
                            3, 1, 1, bias_attr=False),
                        nn.BatchNorm2D(out_channels, momentum=0.1),
                        nn.ReLU(),
                        nn.UpsamplingBilinear2D(scale_factor=2)
                        if num_upsample != 0 else nn.Identity(),
                    ) for idx in range(num_layers)
                ]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)
        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        return out_feat


@manager.MODELS.add_component
class FarSeg(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 fpn_channel_list=(256, 512, 1024, 2048),
                 fpn_out_channel=256,
                 sr_channel_list=(256, 256, 256, 256),
                 sr_out_channel=256,
                 head_out_channel=128,
                 pretrained=None):
        super(FarSeg, self).__init__()

        self.backbone = ResNet50Encoder(in_channels=in_channels,
                                        pretrained=pretrained)
        self.neck = FarSegNeck(fpn_channel_list=fpn_channel_list,
                               fpn_out_channels=fpn_out_channel,
                               sr_in_channels=fpn_channel_list[-1],
                               sr_channel_list=sr_channel_list,
                               sr_out_channels=sr_out_channel,
                               scale_aware_proj=True)
        self.decoder = AsymmetricDecoder(in_channels=sr_out_channel,
                                         out_channels=head_out_channel)
        self.cls_head = nn.Sequential(
            DefaultConvBlock(head_out_channel, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2D(scale_factor=4))

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        feature = self.decoder(features)
        logit = self.cls_head(feature)
        return [logit]
