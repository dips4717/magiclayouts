#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:41:50 2020

@author: dipu
"""

from detectron2.modeling import ROI_HEADS_REGISTRY, ROIHeads
ROI_HEADS_REGISTRY._obj_map.pop('StandardROIHeads_Graph')

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals


from fast_rcnn_comat import FastRCNNOutputLayers_comat, FastRCNNOutputsEnc, FastRCNNOutputLayers_gaussian_comat
from detectron2.modeling.roi_heads.mask_head import build_mask_head


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads_Graph(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads_Graph, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        #self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.head_type    = cfg.MODEL.HEAD_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        if cfg.MODEL.HEAD_TYPE == 'comat':
            self.box_predictor = FastRCNNOutputLayers_comat(
                self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg, comat_file=cfg.comat_file
            )
        elif cfg.MODEL.HEAD_TYPE == 'gaussian_comat':
            self.box_predictor = FastRCNNOutputLayers_gaussian_comat(
                self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg, comat_file=cfg.comat_file
            )
            
        else:
            raise Exception(f'Not Supportted HEAD_TYPE: {cfg.HEAD_TYPE}')

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )    
    
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            #losses.update(self._forward_mask(features, proposals))
            #losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        #instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        if self.head_type  == 'comat':
            #pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
            pred_class_logits, pred_proposal_deltas, pred_class_logits_enc, pred_proposal_deltas_enc = self.box_predictor(box_features)
        elif self.head_type == 'gaussian_comat':
            proposal_boxes = [x.proposal_boxes for x in proposals]
            image_sizes = [x.image_size for x in proposals]
            pred_class_logits, pred_proposal_deltas, pred_class_logits_enc, pred_proposal_deltas_enc = self.box_predictor(box_features, proposal_boxes, image_sizes)
        del box_features

        # outputs = FastRCNNOutputs(
        #     self.box2box_transform,
        #     pred_class_logits,
        #     pred_proposal_deltas,
        #     proposals,
        #     self.smooth_l1_beta,
        # )
        
        outputs = FastRCNNOutputsEnc(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            pred_class_logits_enc,
            pred_proposal_deltas_enc,
            proposals,
            self.smooth_l1_beta,
        )
        
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)   