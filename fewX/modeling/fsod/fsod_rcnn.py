# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Boxes, Instances, pairwise_iou
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .fsod_roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F

from .fsod_fast_rcnn import FsodFastRCNNOutputs

import os

import matplotlib.pyplot as plt
import pandas as pd
from detectron2.layers import ShapeSpec
from detectron2.data.catalog import MetadataCatalog
import detectron2.data.detection_utils as utils
import pickle
import sys

# nested_tensor
sys.path.append("E:\\codes\\FewX-df\\fewx")
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized, inverse_sigmoid)
from utils import box_ops

# backbone部分
sys.path.append("E:\\codes\\FewX-df\\fewx\\modeling\\fsod")
# from transformerrpn.backbone import Backbone, Joiner
from df_detr.position_encoding import PositionEmbeddingSine  # Position encoding
from df_detr.backbone import Backbone
from df_detr.backbone import Joiner

# transformer
from df_detr.deformable_transformer import DeformableTransformer

# rpn loss
from df_detr.matcher import HungarianMatcher
from df_detr.deformable_detr import SetCriterion

__all__ = ["FsodRCNN"]


@META_ARCH_REGISTRY.register()
class FsodRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone) 各个图像的特征提取
    2. Region proposal generation  区域提名产生
    3. Per-region feature extraction and prediction  各个区域的特征提取以及预测
    """

    def __init__(self, cfg):
        super().__init__()
        # 定义关于backbone部分
        '''1.feature abstraction backbone部分
           2.mask下采样部分
           3.position encoding部分'''
        # hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        # N_steps = hidden_dim // 2
        '''train_backbone = 1e-5 > 0
        return_interm_layers = 1 > 0
        dilation = 1 > 0
        N_steps = 256 // 2
        origin_backbone = Backbone('resnet50', train_backbone, return_interm_layers, dilation)
        self.backbone = Joiner(origin_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        self.backbone.num_channels = origin_backbone.num_channels
        device = torch.device('cuda:0')
        self.backbone.to(device)'''

        # 原本的模组定义

        self.cls_backbone = build_backbone(cfg)
        basebackbone = Backbone('resnet50', True, True, False)

        N_steps = 256 // 2
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        self.backbone = Joiner(basebackbone, position_embedding)
        hidden_dim = 256
        num_backbone_outs = len(self.backbone.strides)
        input_proj_list = []
        num_feature_levels = 4
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                # in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)

        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        num_queries = 300


        # self.tgt = MLP(196, hidden_dim, 256, 3)
        # self.tgt = torch.nn.Linear(196, hidden_dim)
        # self.class_embed = torch.nn.Linear(hidden_dim, 1)


        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, 256*2)
        # self.query_embed = nn.Embedding(num_queries, 256*2)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        output_shape = {'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16)}
        self.roi_heads = build_roi_heads(cfg, output_shape)
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.support_way = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot = cfg.INPUT.FS.SUPPORT_SHOT
        self.logger = logging.getLogger(__name__)
        device = torch.device('cuda:0')

        # self.counts = 0
        '''def build_backbone(args):
            position_embedding = build_position_encoding(args)
            train_backbone = args.lr_backbone > 0
            return_interm_layers = args.masks
            backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
            model = Joiner(backbone, position_embedding)
            model.num_channels = backbone.num_channels
            return model'''

        # 定义关于transformer部分

        dropout = 0.1
        nheads = 8
        dim_feedforward = 1024
        enc_layers = 6
        dec_layers = 6

        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            return_intermediate_dec=True,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=False
        )

        num_pred = self.transformer.decoder.num_layers
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        # self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        '''def build_transformer(args):
            return Transformer(
                d_model=args.hidden_dim,
                dropout=args.dropout,
                nhead=args.nheads,
                dim_feedforward=args.dim_feedforward,
                num_encoder_layers=args.enc_layers,
                num_decoder_layers=args.dec_layers,
                normalize_before=args.pre_norm,
                return_intermediate_dec=True,
            )'''

        # hidden_dim = self.transformer.d_model

        # self.input_proj2 = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        # self.downCNN = DownCNN(1024, 100, stride=1)
        # self.downCNN.to(device)

        self.transformer.to(device)



        # 匈牙利算法loss部分 import :HungarianMatcher; SetCriterion
        matcher = HungarianMatcher(cost_bbox=5, cost_giou=2)  # class：1 bbox: 5 giou: 2
        # weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}   mask_loss_coef= 1  dice_loss_coef= 1  bbox_loss_coef= 5  giou_loss_coef= 2  eos_coef= 0.1
        # weight_dict = {'loss_bbox': 5, 'loss_giou': 2, 'loss_ce': 2}
        weight_dict = {'loss_bbox': 2.5, 'loss_giou': 1}
        aux_weight_dict = {}
        for i in range(6 - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        losses = ['boxes']
        self.criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, losses=losses)
        self.criterion.to(device)
        self.count_inference = 0
        self.memory_cor = []

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        # with EventStorage():
        #    storage = get_event_storage()
        max_vis_prop = 20
        support_dir = 'E:\\codes\\FewX-df\\picture'
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            support_file_name_v_gt = os.path.join(support_dir, 'gt', str(self.counts))
            v_gt.save(support_file_name_v_gt)
            # anno_img = v_gt.get_image()
            # box_size = min(len(prop.proposal_boxes), max_vis_prop)
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            '''v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )'''
            v_pred = v_pred.overlay_instances(boxes=prop.proposal_boxes[0:box_size].tensor.clone().detach().cpu().numpy())
            # prop_img = v_pred.get_image()
            # vis_img = np.concatenate((anno_img, prop_img), axis=1)
            # vis_img = prop_img
            # vis_img = vis_img.transpose(2, 0, 1)
            # vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            # vis_name = "Predicted proposals"
            # support_file_name_gt = os.path.join(support_dir, 'gt', str(self.counts))
            support_file_name_pred = os.path.join(support_dir, 'pred', str(self.counts))
            # v_gt.save(support_file_name_gt)
            v_pred.save(support_file_name_pred)
            self.counts += 1
            # storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            self.init_model()
            return self.inference(batched_inputs)
        # 利用nested_tensor获取相同大小的image数据，mask、以及support_images的部分

        # cls_backbone部分
        cls_images, cls_support_images = self.cls_preprocess_image(batched_inputs)
        cls_features = self.cls_backbone(cls_images.tensor)

        images_with_masks, support_images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            for x in batched_inputs:
                x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        images, masks_candidate = images_with_masks.decompose()

        src_out, pos = self.backbone(images_with_masks)  # 利用jointly的backbone获取position encoding
        srcs = []
        srcs_roi = []
        masks = []
        for l, feat in enumerate(src_out):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            srcs_roi.append(src)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):  # num_feature_levels = 4
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](src_out[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks_candidate
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # support branches
        support_bboxes_ls = []
        for item in batched_inputs:
            bboxes = item['support_bboxes']
            for box in bboxes:
                box = Boxes(box[np.newaxis, :])
                support_bboxes_ls.append(box.to(self.device))

        B, N, C, H, W = support_images.tensor.shape
        assert N == self.support_way * self.support_shot
        support_images = support_images.tensor.reshape(B * N, C, H, W)
        support_features = self.cls_backbone(support_images)
        support_box_features = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features],
                                                                    support_bboxes_ls)

        '''support_images_candidate = support_images.tensor.reshape(B * N, C, H, W)
        support_images_nested_candidate = []
        for i in range(B * N):
            support_images_nested_candidate.append(support_images_candidate[i])
        support_images = nested_tensor_from_tensor_list(support_images_nested_candidate)
        # support_images, support_masks_candidate = support_images.decompose()
        # support_features, support_mask = support_features_with_mask[-1].decompose()
        support_features_out, support_features_pos = self.backbone(support_images)
        support_features_srcs = []
        support_features_masks = []
        for l, feat in enumerate(support_features_out):
            src, mask = feat.decompose()
            # support_features_srcs.append(self.input_proj[l](src))
            support_features_srcs.append(src)
            support_features_masks.append(mask)
            assert mask is not None'''
        # support_features = support_features_out['res4']

        # 涉及support mask以及support position encoding的部分

        # support_mask = F.interpolate(support_masks_candidate.float().unsqueeze(0), size=support_features.shape[-2:]).squeeze(0).to(torch.bool)
        # position encoding
        # support_pos_candidate = NestedTensor(support_features, support_mask)
        # support_pos = self.position_embedding(support_pos_candidate).to(support_features.dtype)

        # support feature roi pooling
        # feature_pooled = self.roi_heads.roi_pooling(support_features, support_bboxes_ls)

        # support_box_features = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features], support_bboxes_ls)
        # support_box_features = self.roi_heads._shared_roi_transform([support_features], support_bboxes_ls)
        # support_features_last = support_features_srcs[-1]
        # tgt_input_candidate = self.roi_heads.roi_pooling([support_features_srcs[1]], support_bboxes_ls)  # [80,1024,20,20]-->[80,1024,14,14]
        '''support_box_features = self.roi_heads._shared_roi_transform([support_features_srcs[1]],
                                                                    support_bboxes_ls)  # [80,1024,20,20]-->[80,2048,7,7]'''
        # support_box_features = tgt_input_candidate
        # assert self.support_way == 2  # now only 2 way support

        detector_loss_cls = []
        detector_loss_box_reg = []
        # rpn_loss_rpn_obj = []
        rpn_loss_rpn_loc = []
        for i in range(B):  # batch
            # query
            query_gt_instances = [gt_instances[i]]  # one query gt instances
            #query_images = ImageList.from_tensors([images[i]])  # one query image

            # cls输入准备
            cls_query_images = ImageList.from_tensors([cls_images[i]])  # one query image
            cls_query_feature_res4 = cls_features['res4'][i].unsqueeze(0)  # one query feature for attention rpn
            # cls_query_features = {'res4': cls_query_feature_res4}  # one query feature for rcnn

            # Transformer_RPN pos输入准备
            query_feature_res4 = []
            query_feature_roi = srcs_roi[1][i].unsqueeze(0)
            query_mask = []
            pos_embed = []
            for a in range(len(srcs)):
                query_feature_res4.append(
                    srcs[a][i].unsqueeze(0))  # one query feature for attention rpn [1,1024,36,48]   ########inf
                # query_feature_roi.append(srcs_roi[a][i].unsqueeze(0))
                query_mask.append(masks[a][i].unsqueeze(0))  # [1,36,48]
                pos_embed.append(pos[a][i].unsqueeze(0))  # [1,256,36,48]  ##############
            query_embeds = self.query_embed.weight  # [50,196]

            # query_features = {'res4': query_feature_res4}  # one query feature for rcnn

            # positive support branch ##################################
            pos_begin = i * self.support_shot * self.support_way
            pos_end = pos_begin + self.support_shot
            # pos_support_features = feature_pooled[pos_begin:pos_end].mean(0, True) # pos support features from res4, average all supports, for rcnn
            # pos_support_features_pool = pos_support_features.mean(dim=[2, 3], keepdim=True) # average pooling support feature for attention rpn
            # pos_correlation = F.conv2d(query_feature_res4, pos_support_features_pool.permute(1,0,2,3), groups=1024) # attention map

            # pos_features = {'res4': pos_correlation} # attention map for attention rpn
            # pos_support_features = tgt_input_candidate[pos_begin:pos_end].mean(0, True)  # [1,1024,14,14]
            pos_support_box_features = support_box_features[pos_begin:pos_end].mean(0, True)  # 输入的BOX query [1,2048,7,7]   ########没问题
            # pos_tgt_input_candidate = self.downCNN(pos_support_features).flatten(2)  # #####################[1,1024,196]
            # pos_tgt_input = self.tgt(pos_tgt_input_candidate)
            # pos_proposals, pos_anchors, pos_pred_objectness_logits, pos_gt_labels, pos_pred_anchor_deltas, pos_gt_boxes = self.proposal_generator(query_images, pos_features, query_gt_instances) # attention rpn
            '''hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
                query_feature_res4, query_mask, pos_embed, pos_tgt_input, query_embeds)'''
            hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
                query_feature_res4, query_mask, pos_embed,  query_embeds)

            # Transformer_RPN处理(pos)
            # pos_proposals_candidate = self.Transformer_RPN(self.input_proj2(query_feature_res4), query_mask, query_embed, pos_embed, pos_tgt_input)[0][5]  # pos_support_box_featires是为query_embed
            # pos_proposals = self.bbox_embed(pos_proposals_candidate).sigmoid() # pos_proposals_candidate [6,1,100,196] pos_proposals [6,1,100,4]
            # pos_objectness_logits = torch.tanh(self.objectness_logits(pos_proposals_candidate))

            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                #outputs_class = torch.sigmoid(self.class_embed[lvl](hs[lvl]))
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                # outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            #outputs_class = torch.stack(outputs_classes)  #[6,1,500,1]
            outputs_coord = torch.stack(outputs_coords)

            # output_class_obj = outputs_class.mean(0)

            #_, indices = torch.sort(outputs_class[-1].squeeze(0).squeeze(1), descending=True)  # descending：递减（从大到小）

            for j in range(outputs_coord.size()[0]):
                outputs_coord[j] = box_ops.box_cxcywh_to_xyxy(outputs_coord[j])
            # outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
            #            'aux_outputs': [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]}
            outputs = {'pred_boxes': outputs_coord[-1], 'aux_outputs': [{'pred_boxes': a} for a in outputs_coord[:-1]]}

            pos_ins = Instances((batched_inputs[i]['image'].size()[-2], batched_inputs[i]['image'].size()[-1]))  # 在origin里是数据增强后的扩大尺寸，非统一扩大尺寸
            # print(outputs_coord[-1].squeeze(0).size())
            # pos_ins.proposal_boxes = Boxes(size_transform(outputs_coord[-1].squeeze(0)[indices][:100], images, i))  # 由于是用于提取roi特征的，而roi特征是基于统一扩大尺寸后得到的特征图的，故此处的坐标应为统一扩大后坐标
            pos_ins.proposal_boxes = Boxes(size_transform(outputs_coord[-1].squeeze(0), images, i))
            # print(pos_ins.proposal_boxes,query_gt_instances[0].gt_boxes.tensor )

            # 可视化部分
            '''visdict = {'image': images[0], 'instances': batched_inputs[0]['instances']}
            self.visualize_training([visdict], [pos_ins])'''

            # multi_relation处理
            pos_pred_class_logits, pos_pred_proposal_deltas, pos_detector_proposals = self.roi_heads(cls_query_images,
                                                                                                     cls_query_feature_res4,
                                                                                                     pos_support_box_features,
                                                                                                     [pos_ins],
                                                                                                     query_gt_instances)  # pos rcnn
            # rpn loss
            # 此处的归一化后尺寸应为对应统一扩大尺寸的数值，因为预测是是基于统一扩大尺寸的，则目标值也应是基于统一扩大尺寸的
            # 注意，此处的归一化尺寸应为H,W，别错判为X，Y，X->W,Y->H
            target_templet = torch.tensor([[images[i].size()[-1],
                                            images[i].size()[-2],
                                            images[i].size()[-1],
                                            images[i].size()[-2]]]).to('cuda:0')
            targets_boxes = query_gt_instances[0].gt_boxes.tensor / target_templet

            # targets = [{'boxes': targets_boxes, 'labels': torch.ones(len(targets_boxes))}]  # exp: query_gt_instances.gt_boxes [1,4]
            targets = [{'boxes': targets_boxes}]
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict  # 各项loss的权重系数
            # storage = get_event_storage()
            # weight_dict = {'loss_bbox': 0.5, 'loss_giou': 0.2}
            '''if storage.iter > 10000:
                weight_dict['loss_bbox'] = 0.25
                weight_dict['loss_giou'] = 0.1
            if storage.iter > 20000:
                weight_dict['loss_bbox'] = 0.125
                weight_dict['loss_giou'] = 0.05
            if storage.iter > 30000:
                weight_dict['loss_bbox'] = 0.0625
                weight_dict['loss_giou'] = 0.025
            if storage.iter > 40000:
                weight_dict['loss_bbox'] = 0.03125
                weight_dict['loss_giou'] = 0.0125'''
            # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            proposal_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # detector loss
            detector_pred_class_logits = pos_pred_class_logits
            detector_pred_proposal_deltas = pos_pred_proposal_deltas
            # for item in neg_detector_proposals:
            #    item.gt_classes = torch.full_like(item.gt_classes, 1)

            # detector_proposals = pos_detector_proposals + neg_detector_proposals
            detector_proposals = pos_detector_proposals
            if self.training:
                predictions = detector_pred_class_logits, detector_pred_proposal_deltas
                detector_losses = self.roi_heads.box_predictor.losses(predictions, detector_proposals)

            # rpn_loss_rpn_obj.append(objectness_loss)
            rpn_loss_rpn_loc.append(proposal_losses)
            detector_loss_cls.append(detector_losses['loss_cls'])
            detector_loss_box_reg.append(detector_losses['loss_box_reg'])
            # print(detector_losses)

        proposal_losses = {}
        detector_losses = {}

        # proposal_losses['loss_rpn_obj'] = torch.stack(rpn_loss_rpn_obj).mean()
        proposal_losses['loss_rpn_loc'] = torch.stack(rpn_loss_rpn_loc).mean()
        detector_losses['loss_cls'] = torch.stack(detector_loss_cls).mean()
        detector_losses['loss_box_reg'] = torch.stack(detector_loss_box_reg).mean()

        losses = {}
        # losses.update(proposal_losses)
        # storage = get_event_storage()
        # if storage.iter > 10000:
        losses.update(detector_losses)

        return losses

    def init_model(self):
        self.support_on = True  # False

        support_dir = './support_dir'
        if not os.path.exists(support_dir):
            os.makedirs(support_dir)

        support_file_name = os.path.join(support_dir, 'support_feature.pkl')
        if not os.path.exists(support_file_name):
            support_path = './datasets/coco/10_shot_support_df.pkl'
            support_df = pd.read_pickle(support_path)

            metadata = MetadataCatalog.get('coco_2017_train')
            # unmap the category mapping ids for COCO
            reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
            support_df['category_id'] = support_df['category_id'].map(reverse_id_mapper)

            support_dict = {'res4_avg': {}, 'res5_avg': {}}
            #support_dict = {'res4_avg': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join('./datasets/coco', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images)  #### N,C,H,W
                support_features = self.cls_backbone(support_images.tensor)

                '''N, C, H, W = support_images.tensor.shape
                support_images_nested_candidate = []
                for i in range(N):
                    support_images_nested_candidate.append(support_images[i])
                support_images = nested_tensor_from_tensor_list(support_images_nested_candidate)
                support_features_out, support_features_pos = self.backbone(support_images)
                support_features_srcs = []
                support_features_masks = []
                for l, feat in enumerate(support_features_out):
                    src, mask = feat.decompose()
                    # support_features_srcs.append(self.input_proj[l](src))
                    support_features_srcs.append(src)
                    support_features_masks.append(mask)
                    assert mask is not None

                res4_pooled = self.roi_heads.roi_pooling([support_features_srcs[1]],
                                                         support_box_all)'''

                res4_pooled = self.roi_heads.roi_pooling(support_features, support_box_all)

                res4_avg = res4_pooled.mean(0, True)
                # res4_avg = res4_avg.mean(dim=[2,3], keepdim=True)
                support_dict['res4_avg'][cls] = res4_avg.detach().cpu().data

                '''tgt_feature = self.roi_heads.roi_pooling([support_features_srcs[1]], support_box_all)
                tgt_avg = tgt_feature.mean(0, True)
                support_dict['tgt_avg'][cls] = tgt_avg.detach().cpu().data'''

                # res5_feature = self.roi_heads._shared_roi_transform([support_features_srcs[1]], support_box_all)

                res5_feature = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features],
                                                                    support_box_all)
                res5_avg = res5_feature.mean(0, True)
                support_dict['res5_avg'][cls] = res5_avg.detach().cpu().data

                '''res5_feature = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features], support_box_all)
                res5_avg = res5_feature.mean(0, True)
                support_dict['res5_avg'][cls] = res5_avg.detach().cpu().data'''

                del res4_avg
                del res4_pooled
                #del support_features_srcs
                del support_features
                del res5_feature
                del res5_avg

            with open(support_file_name, 'wb') as f:
                pickle.dump(support_dict, f)
            self.logger.info("=========== Offline support features are generated. ===========")
            self.logger.info("============ Few-shot object detetion will start. =============")
            sys.exit(0)

        else:
            with open(support_file_name, "rb") as hFile:
                self.support_dict = pickle.load(hFile, encoding="latin1")
            for res_key, res_dict in self.support_dict.items():
                for cls_key, feature in res_dict.items():
                    self.support_dict[res_key][cls_key] = feature.cuda()

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images_with_masks = self.preprocess_image(batched_inputs)

        cls_images = self.cls_preprocess_image(batched_inputs)
        cls_features = self.cls_backbone(cls_images.tensor)

        images, masks_candidate = images_with_masks.decompose()

        src_out, pos = self.backbone(images_with_masks)  # 利用jointly的backbone获取position encoding
        srcs = []
        srcs_roi = []
        masks = []
        for l, feat in enumerate(src_out):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            srcs_roi.append(src)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):  # num_feature_levels = 4
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](src_out[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks_candidate
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        B, _, _, _ = srcs[1].shape
        assert B == 1  # only support 1 query image in test
        # assert len(images) == 1
        support_proposals_dict = {}
        support_box_features_dict = {}
        proposal_num_dict = {}

        for cls_id, res4_avg in self.support_dict['res4_avg'].items():
            query_images = ImageList.from_tensors([images[0]])  # one query image

            cls_query_features_res4 = cls_features['res4']  # one query feature for attention rpn
            # cls_query_features = {'res4': cls_query_features_res4}

            query_feature_res4 = []
            query_feature_roi = srcs_roi[1][0].unsqueeze(0)
            query_mask = []
            pos_embed = []
            for a in range(len(srcs)):
                query_feature_res4.append(
                    srcs[a][0].unsqueeze(0))  # one query feature for attention rpn [1,1024,36,48]   ########inf
                # query_feature_roi.append(srcs_roi[a][i].unsqueeze(0))
                query_mask.append(masks[a][0].unsqueeze(0))  # [1,36,48]
                pos_embed.append(pos[a][0].unsqueeze(0))  # [1,256,36,48]  ##############
            query_embeds = self.query_embed.weight  # [50,196]
            # pos_support_features = res4_avg  # [1,1024,1,1]
            # pos_tgt_input_candidate = self.downCNN(pos_support_features).flatten(2)  # #####################[1,1024,196]
            # pos_tgt_input = self.tgt(pos_tgt_input_candidate)

            '''hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
                query_feature_res4, query_mask, pos_embed, pos_tgt_input, query_embeds)'''
            hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
                query_feature_res4, query_mask, pos_embed, query_embeds)

            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

            for j in range(outputs_coord.size()[0]):
                outputs_coord[j] = box_ops.box_cxcywh_to_xyxy(outputs_coord[j])

            '''pos_proposals_final = outputs_coord[-1].squeeze(0).squeeze(0)  # [100,4]
            pos_proposals_final_transform = box_ops.box_cxcywh_to_xyxy(pos_proposals_final)
            pos_transform_templete = torch.zeros_like(pos_proposals_final)
            pos_transform_templete[:, 0] = query_images.image_sizes[0][0]
            pos_transform_templete[:, 2] = query_images.image_sizes[0][0]
            pos_transform_templete[:, 1] = query_images.image_sizes[0][1]
            pos_transform_templete[:, 3] = query_images.image_sizes[0][1]
            pos_proposals_final_transform_final = pos_proposals_final_transform * pos_transform_templete'''

            pos_ins = Instances((batched_inputs[0]['image'].size()[-2], batched_inputs[0]['image'].size()[-1]))
            pos_ins.proposal_boxes = Boxes(size_transform(outputs_coord[-1].squeeze(0), images, 0))
            # pos_ins.objectness_logits = pos_objectness_logits.squeeze(0).squeeze(0).squeeze(1)


            support_proposals_dict[cls_id] = [pos_ins]
            support_box_features_dict[cls_id] = self.support_dict['res5_avg'][cls_id]
            #support_box_features_dict[cls_id] = self.support_dict['res4_avg'][cls_id]

            if cls_id not in proposal_num_dict.keys():
                proposal_num_dict[cls_id] = []
            proposal_num_dict[cls_id].append(len(pos_ins))

            # del pos_support_features
            # del correlation
            del res4_avg
            del query_feature_res4
        # print(batched_inputs[0]['file_name'])
        self.memory_cor.append(support_proposals_dict)
        if self.count_inference % 10 == 0:
            print(self.count_inference)
        if self.count_inference == 4999:
            torch.save(self.memory_cor, './memory_cor.pth')
        if self.count_inference == 5000:
            print('5000 is found')
        self.count_inference += 1

        results, _ = self.roi_heads.eval_with_support(query_images, cls_query_features_res4, support_proposals_dict,
                                                      support_box_features_dict)
        # self.visualize_training(batched_inputs, results)
        image_sizes = [(images[0].shape[-2], images[0].shape[-1])]
        if do_postprocess:
            return FsodRCNN._postprocess(results, batched_inputs, image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_with_mask = nested_tensor_from_tensor_list(images)
        # images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        if self.training:
            # support images
            support_images = [x['support_images'].to(self.device) for x in batched_inputs]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            # support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
            support_images = ImageList.from_tensors(support_images)

            return images_with_mask, support_images
        else:
            return images_with_mask

    def cls_preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.cls_backbone.size_divisibility)
        if self.training:
            # support images
            support_images = [x['support_images'].to(self.device) for x in batched_inputs]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            support_images = ImageList.from_tensors(support_images, self.cls_backbone.size_divisibility)

            return images, support_images
        else:
            return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        # 此处第一次用了实际图片的尺寸（JPG文件的属性尺寸）  一共有三种尺寸：1.JPG尺寸 只有此处用了  2.data argumentation尺寸，image_size中的尺寸，bbox GT以及preprocessing之前用了  3.preprocessing尺寸，也即Tensor尺寸，网络中使用。
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def size_transform(original, batched_inputs, batch_iter):
    transform_templete = torch.zeros_like(original)
    # pos_transform_templete[:, 0] = query_gt_instances[0].image_size[0]
    # print(batched_inputs[i]['image'].size()[-2],batched_inputs[i]['image'].size()[-1])
    transform_templete[:, 0] = batched_inputs[batch_iter].size()[-1]
    transform_templete[:, 2] = batched_inputs[batch_iter].size()[-1]
    transform_templete[:, 1] = batched_inputs[batch_iter].size()[-2]
    transform_templete[:, 3] = batched_inputs[batch_iter].size()[-2]
    # transformed = box_ops.box_cxcywh_to_xyxy(original)

    return original * transform_templete


class DownCNN(nn.Module):
    def __init__(self, in_places, places, stride=1):
        super(DownCNN, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=in_places//4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_places//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_places//4, out_channels=in_places//4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_places//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_places//4, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
        )
        self.conv = nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False)
        self.BN = nn.BatchNorm2d(places)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.BN(self.conv(x))
        out = self.bottleneck(x)

        out += residual
        out = self.relu(out)
        return out

def load_backbone(backbone):
    pretrained_dict = torch.load('E:\\codes\\FewX-df\\resnet50-19c8e357.pth')
    model_dict = backbone.state_dict()

    print('backbone随机初始化权重第一层：', model_dict['backbone.0.body.conv1.weight'])

    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    print('backbone预训练权重第一层：', pretrained_dict['backbone.0.body.conv1.weight'])

    # 更新现有的model_dict
    model_dict.update(pretrained_dict)  # 利用预训练模型的参数，更新模型
    backbone.load_state_dict(model_dict)

    print('backbone更新后权重第一层：', model_dict['backbone.0.body.conv1.weight'])

def load_TFPN(tfpn, query_embed, input_proj, bbox_embed):
    pretrained_dict = torch.load('E:\\codes\\FewX-df\\r50_deformable_detr-checkpoint.pth')['model']
    tfpn_dict = tfpn.state_dict()
    query_embed_dict = query_embed.state_dict()
    input_proj_dict = input_proj.state_dict()
    bbox_embed_dict = bbox_embed.state_dict()

    print('tfpn随机初始化权重第一层：', tfpn_dict['transformer.level_embed'])


    # 将pretrained_dict里不属于model_dict的键剔除掉
    tfpn_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in tfpn_dict}
    query_embed_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in query_embed_dict}
    input_proj_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in input_proj_dict}
    bbox_embed_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in bbox_embed_dict}

    print('tfpn预训练权重第一层：', tfpn_pretrained_dict['transformer.level_embed'])

    # 更新现有的model_dict
    tfpn_dict.update(tfpn_pretrained_dict)  # 利用预训练模型的参数，更新模型
    tfpn.load_state_dict(tfpn_dict)

    query_embed_dict.update(query_embed_pretrained_dict)
    query_embed.load_state_dict(query_embed_dict)

    input_proj_dict.update(input_proj_pretrained_dict)
    input_proj.load_state_dict(input_proj_dict)

    bbox_embed_dict.update(bbox_embed_pretrained_dict)
    bbox_embed.load_state_dict(bbox_embed_dict)

    print('tfpn更新后权重第一层：', tfpn_dict['transformer.level_embed'])
