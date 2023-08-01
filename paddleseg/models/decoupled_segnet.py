# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.backbones import resnet_vd
from paddleseg.models import deeplab
from paddleseg.utils import utils
from .gscnn import GSCNNHead, ASPPModule, GatedSpatailConv2d


@manager.MODELS.add_component
class DecoupledSegNet(nn.Layer):
    """
    The DecoupledSegNet implementation based on PaddlePaddle.

    The original article refers to
    Xiangtai Li, et, al. "Improving Semantic Segmentation via Decoupled Body and Edge Supervision"
    (https://arxiv.org/pdf/2007.10035.pdf)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        backbone_channels = self.backbone.feat_channels
        self.head = DecoupledSegNetHead(num_classes, backbone_indices,
                                        backbone_channels, aspp_ratios,
                                        aspp_out_channels, align_corners)

        # self.aspp = ASPPModule(
        #     aspp_ratios=aspp_ratios,
        #     in_channels=backbone_channels[-1],
        #     out_channels=aspp_out_channels,
        #     align_corners=self.align_corners,
        #     image_pooling=True)
        #
        # self.decoder = deeplab.Decoder(
        #     num_classes=num_classes,
        #     in_channels=backbone_channels[0],
        #     align_corners=self.align_corners)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)

        logit_list = self.head(x, feat_list, self.backbone.conv1_logit)

        seg_logit, body_logit, edge_logit, gscnn_edge_out = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

        if self.training:
            return [seg_logit, body_logit, edge_logit, (seg_logit, edge_logit), seg_logit, gscnn_edge_out]

        return [seg_logit]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class DecoupledSegNetHead(nn.Layer):
    """
    The DecoupledSegNetHead implementation based on PaddlePaddle.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Edge presevation component;
            the second one will be taken as input of ASPP component.
        backbone_channels (tuple): The channels of output of backbone.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, aspp_out_channels, align_corners):
        super().__init__()
        self.backbone_indices = backbone_indices
        self.align_corners = align_corners
        self.aspp = ASPPModule(
            aspp_ratios=aspp_ratios,
            in_channels=backbone_channels[backbone_indices[1]],
            out_channels=aspp_out_channels,
            align_corners=align_corners,
            image_pooling=True)

        self.bot_fine = nn.Conv2D(
            backbone_channels[backbone_indices[0]], 48, 1, bias_attr=False)
        # decoupled
        self.squeeze_body_edge = SqueezeBodyEdge(
            256, align_corners=self.align_corners)
        self.edge_fusion = nn.Conv2D(256 + 48, 256, 1, bias_attr=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=256,
                out_channels=48,
                kernel_size=3,
                bias_attr=False),
            nn.Conv2D(
                48, 1, 1, bias_attr=False))
        self.dsn_seg_body = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                bias_attr=False),
            nn.Conv2D(
                256, num_classes, 1, bias_attr=False))

        self.final_seg = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                bias_attr=False),
            layers.ConvBNReLU(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                bias_attr=False),
            nn.Conv2D(
                256, num_classes, kernel_size=1, bias_attr=False))

        # GSCNN
        self.dsn1 = nn.Conv2D(
            backbone_channels[1], 1, kernel_size=1)
        self.dsn2 = nn.Conv2D(
            backbone_channels[2], 1, kernel_size=1)
        self.dsn3 = nn.Conv2D(
            backbone_channels[3], 1, kernel_size=1)

        self.res1 = resnet_vd.BasicBlock(64, 64, stride=1)
        self.d1 = nn.Conv2D(64, 32, kernel_size=1)
        self.gate1 = GatedSpatailConv2d(32, 32)
        self.res2 = resnet_vd.BasicBlock(32, 32, stride=1)
        self.d2 = nn.Conv2D(32, 16, kernel_size=1)
        self.gate2 = GatedSpatailConv2d(16, 16)
        self.res3 = resnet_vd.BasicBlock(16, 16, stride=1)
        self.d3 = nn.Conv2D(16, 8, kernel_size=1)
        self.gate3 = GatedSpatailConv2d(8, 8)
        self.fuse = nn.Conv2D(8, 1, kernel_size=1, bias_attr=False)

        self.cw = nn.Conv2D(2, 1, kernel_size=1, bias_attr=False)

        self.edge_conv = layers.ConvBNReLU(
            1, 48, kernel_size=1, bias_attr=False)

    def forward(self, x, feat_list, mlf):
        fine_fea = feat_list[self.backbone_indices[0]]
        fine_size = paddle.shape(fine_fea)

        l1, l2, l3= [
            feat_list[i]
            for i in range(1, 4)
        ]
        s1 = F.interpolate(
            self.dsn1(l1),
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        s2 = F.interpolate(
            self.dsn2(l2),
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        s3 = F.interpolate(
            self.dsn3(l3),
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # Get image gradient
        input_shape = paddle.shape(x)
        im_arr = x.numpy().transpose((0, 2, 3, 1))
        im_arr = ((im_arr * 0.5 + 0.5) * 255).astype(np.uint8)
        canny = np.zeros((input_shape[0], 1, input_shape[2], input_shape[3]))
        for i in range(input_shape[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = F.interpolate(
                paddle.to_tensor(canny),
                fine_size[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        canny = canny / 255
        canny = paddle.to_tensor(canny).astype('float32')
        canny.stop_gradient = True

        cs = self.res1(mlf)
        cs = F.interpolate(
            cs,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cs = self.d1(cs)
        cs = self.gate1(cs, s1)

        cs = self.res2(cs)
        cs = F.interpolate(
            cs,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cs = self.d2(cs)
        cs = self.gate2(cs, s2)

        cs = self.res3(cs)
        cs = F.interpolate(
            cs,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cs = self.d3(cs)
        cs = self.gate3(cs, s3)
        cs = self.fuse(cs)
        cs = F.interpolate(
            cs,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        gscnn_edge_out = F.sigmoid(cs)

        cat = paddle.concat([gscnn_edge_out, canny], axis=1)
        acts = self.cw(cat)
        acts = F.sigmoid(acts)

        # Decouple Seg Net
        x = feat_list[self.backbone_indices[1]]
        aspp = self.aspp(x, acts)

        # decoupled
        seg_body, seg_edge = self.squeeze_body_edge(aspp)
        # Edge presevation and edge out
        fine_fea = self.bot_fine(fine_fea)
        seg_edge = F.interpolate(
            seg_edge,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_edge = self.edge_fusion(paddle.concat([seg_edge, fine_fea], axis=1))
        seg_edge_out = self.edge_out(seg_edge)
        seg_edge_out = self.sigmoid_edge(seg_edge_out)  # seg_edge output
        seg_body_out = self.dsn_seg_body(seg_body)  # body out

        # seg_final out
        seg_out = seg_edge + F.interpolate(
            seg_body,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        aspp = F.interpolate(
            aspp,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_out = paddle.concat([aspp, seg_out], axis=1)
        seg_final_out = self.final_seg(seg_out)

        return [seg_final_out, seg_body_out, seg_edge_out, gscnn_edge_out]


class SqueezeBodyEdge(nn.Layer):
    def __init__(self, inplane, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.down = nn.Sequential(
            layers.ConvBNReLU(
                inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            layers.ConvBNReLU(
                inplane, inplane, kernel_size=3, groups=inplane, stride=2))
        self.flow_make = nn.Conv2D(
            inplane * 2, 2, kernel_size=3, padding='same', bias_attr=False)

    def forward(self, x):
        size = paddle.shape(x)[2:]
        seg_down = self.down(x)
        seg_down = F.interpolate(
            seg_down,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        flow = self.flow_make(paddle.concat([x, seg_down], axis=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        input_shape = paddle.shape(input)
        norm = size[::-1].reshape([1, 1, 1, -1])
        norm.stop_gradient = True
        h_grid = paddle.linspace(-1.0, 1.0, size[0]).reshape([-1, 1])
        h_grid = h_grid.tile([size[1]])
        w_grid = paddle.linspace(-1.0, 1.0, size[1]).reshape([-1, 1])
        w_grid = w_grid.tile([size[0]]).transpose([1, 0])
        grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)
        grid.unsqueeze(0).tile([input_shape[0], 1, 1, 1])
        grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

        output = F.grid_sample(input, grid)
        return output
