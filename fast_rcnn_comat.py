# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Linear, batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import pickle
import torch.distributions as tdist

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"
            self.image_shapes = [x.image_size for x in proposals]

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
            self.image_shapes = []
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * F.cross_entropy(
                self.pred_class_logits,
                torch.zeros(0, dtype=torch.long, device=self.pred_class_logits.device),
                reduction="sum",
            )
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * smooth_l1_loss(
                self.pred_proposal_deltas,
                torch.zeros_like(self.pred_proposal_deltas),
                0.0,
                reduction="sum",
            )
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regionspred_proposal_deltas
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.0
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        # If the box head is class-agnostic, then the method is equivalent to `predicted_boxes`.
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(num_pred, num_classes, B)[
                torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes
            ]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

class FastRCNNOutputsEnc(object):
    """
    Same as FastRCNNOutputs but takes in additional pred_class_logits_enc and pred_proposal_deltas_enc for enchanced preduictions
    Use this to compute losses: 2-classification losses, 2-bbox pred losses

    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, pred_class_logits_enc, pred_proposal_deltas_enc, proposals, smooth_l1_beta
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.pred_class_logits_enc = pred_class_logits_enc
        self.pred_proposal_deltas_enc = pred_proposal_deltas_enc
        self.smooth_l1_beta = smooth_l1_beta

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"
            self.image_shapes = [x.image_size for x in proposals]

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
            self.image_shapes = []
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            cls_loss = 0.0 * F.cross_entropy(
                self.pred_class_logits,
                torch.zeros(0, dtype=torch.long, device=self.pred_class_logits.device),
                reduction="sum",
            )
            cls_loss_enc = 0.0 * F.cross_entropy(
                self.pred_class_logits_enc,
                torch.zeros(0, dtype=torch.long, device=self.pred_class_logits.device),
                reduction="sum",
            )
            return cls_loss, cls_loss_enc
        else:
            self._log_accuracy()
            cls_loss = F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")
            cls_loss_enc = F.cross_entropy(self.pred_class_logits_enc, self.gt_classes, reduction="mean")
            return cls_loss, cls_loss_enc

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            loss_box_reg = 0.0 * smooth_l1_loss(
                self.pred_proposal_deltas,
                torch.zeros_like(self.pred_proposal_deltas),
                0.0,
                reduction="sum",
            )
            
            loss_box_reg_enc = 0.0 * smooth_l1_loss(
                self.pred_proposal_deltas_enc,
                torch.zeros_like(self.pred_proposal_deltas_enc),
                0.0,
                reduction="sum",
            )
            
            return loss_box_reg, loss_box_reg_enc
        
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        loss_box_reg_enc = smooth_l1_loss(
            self.pred_proposal_deltas_enc[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regionspred_proposal_deltas
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        loss_box_reg_enc = loss_box_reg_enc / self.gt_classes.numel()
        return loss_box_reg, loss_box_reg_enc

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        #K = self.pred_proposal_deltas.shape[1] // B
        K = self.pred_proposal_deltas_enc.shape[1] // B
        #boxes = self.box2box_transform.apply_deltas(
        #    self.pred_proposal_deltas.view(num_pred * K, B),
        #    self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        #)
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas_enc.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.0
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        cls_loss, cls_loss_enc = self.softmax_cross_entropy_loss()
        loss_box_reg, loss_box_reg_enc = self.smooth_l1_loss()
        
        return {
            "loss_cls": cls_loss,
            "loss_box_reg": loss_box_reg,
            "loss_cls_enc": cls_loss_enc,
            "loss_box_reg_enc": loss_box_reg_enc            
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        # If the box head is class-agnostic, then the method is equivalent to `predicted_boxes`.
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(num_pred, num_classes, B)[
                torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes
            ]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        #probs = F.softmax(self.pred_class_logits, dim=-1)
        probs = F.softmax(self.pred_class_logits_enc, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


class FastRCNNOutputLayers_comat(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, comat_file=None):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers_comat, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        
        graph_output_size = 512
        self.cls_score_enc = Linear(input_size + graph_output_size, num_classes + 1)
        self.bbox_pred_enc = Linear(input_size + graph_output_size, num_bbox_reg_classes * box_dim)
        
        nn.init.normal_(self.cls_score_enc.weight, std=0.01)
        nn.init.normal_(self.bbox_pred_enc.weight, std=0.001)
        
        for l in [self.cls_score, self.bbox_pred, self.cls_score_enc, self.bbox_pred_enc]:
            nn.init.constant_(l.bias, 0)
            
        
        #adj_gt = '/home/dipu/codes/detectron2/rico/data/comat_overlap_h10_dtrn.pkl'
        if comat_file == '':
            self.adj_gt = torch.rand(num_classes + 1, num_classes + 1)   
            self.adj_gt = self.adj_gt.cuda()
        else:
            adj_gt = comat_file
            print('\n Using co-occ matrix from {} \n'.format(comat_file))
            adj_gt = pickle.load(open(adj_gt, 'rb'))
            adj_gt = np.float32(adj_gt)
            self.adj_gt = nn.Parameter(torch.from_numpy(adj_gt), requires_grad=False)


        #self.graph_fc = Linear(1025, graph_output_size)  # For DC5 and FPN
        #self.graph_fc = Linear(2049, graph_output_size)  # For C4
        self.graph_fc = Linear(input_size +1, graph_output_size) 
        
    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        
        #
        W_classifier = torch.cat((self.cls_score.weight, self.cls_score.bias.unsqueeze(1)),1).detach()  # Cx1025
        
        em = torch.mm(self.adj_gt, W_classifier) # (CxC) X (Cx1025)
        em_fc = self.graph_fc(em)  #Cx512
        em_fc = F.relu(em_fc)   
        
        # Class_probs/ Soft-mapping/ NrxC
        Ms =  nn.Softmax(1)(scores).detach()  # detach/no  Num_rois_per_head*ims_per_batch X C 
        enc_feat = torch.mm(Ms, em_fc)     # 128*2 X 1025
           
        enc_feat = torch.cat((enc_feat, x), dim=1)
        
        scores_enc = self.cls_score_enc(enc_feat)   # 
        proposal_deltas_enc = self.bbox_pred_enc(enc_feat)
        
        return scores, proposal_deltas, scores_enc, proposal_deltas_enc
    
    
    
    
class FastRCNNOutputLayers_gaussian_comat(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, comat_file=None):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers_gaussian_comat, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        
        graph_output_size = 512
        self.cls_score_enc = Linear(input_size + graph_output_size, num_classes + 1)
        self.bbox_pred_enc = Linear(input_size + graph_output_size, num_bbox_reg_classes * box_dim)
        
        nn.init.normal_(self.cls_score_enc.weight, std=0.01)
        nn.init.normal_(self.bbox_pred_enc.weight, std=0.001)
        
        for l in [self.cls_score, self.bbox_pred, self.cls_score_enc, self.bbox_pred_enc]:
            nn.init.constant_(l.bias, 0)
            
        
        #adj_gt = '/home/dipu/codes/detectron2/rico/data/comat_overlap_h10_dtrn.pkl'
        if comat_file == '':
            self.co_mats = torch.rand(10,2,2).cuda()
            
        else:
            adj_gt = comat_file
            print('\n Using co-occ matrix from {} \n'.format(comat_file))
            co_mats = pickle.load(open(adj_gt, 'rb'))
            co_mats = np.stack([v for k,v in co_mats.items()])
            co_mats = np.float32(co_mats)
            self.co_mats = nn.Parameter(torch.from_numpy(co_mats), requires_grad=False)
        self.gaussian = tdist.Normal(torch.tensor([0.0], device='cuda'), torch.tensor([0.3], device='cuda'))
        ycent = torch.arange(0.05,1,0.1)
        self.ycent = nn.Parameter(ycent, requires_grad=False)
        
        #comat = {k: np.float32(v) for k,v in adj_gt.items()}  # np.float32(adj_gt)
        #self.comat = {k: nn.Parameter(torch.from_numpy(v), requires_grad=False) for k,v in comat.items()}  # nn.Parameter(torch.from_numpy(adj_gt), requires_grad=False)
        
        # self.adj_gt = torch.rand(num_classes + 1, num_classes + 1)   
        # self.adj_gt = self.adj_gt.cuda()
        
        #self.graph_fc = Linear(1025, graph_output_size)  # For DC5 and FPN
        #self.graph_fc = Linear(2049, graph_output_size)  # For C4
        self.graph_fc = Linear(input_size +1, graph_output_size) 
        
    def forward(self, x, proposals, image_sizes):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        
        num_preds_per_image = [len(p) for p in proposals]
        
        gaussian_wts = []
        for i, (tmp_proposals, temp_img_size) in  enumerate(zip(proposals,image_sizes)):
            rois = tmp_proposals.tensor
            ycen_prop =  (rois[:,1] + rois[:,3])/2
            ycen_prop = ycen_prop/temp_img_size[1]
            dist =   self.ycent.unsqueeze(0) - ycen_prop.unsqueeze(1)
            wts = torch.exp(self.gaussian.log_prob(dist))
            gaussian_wts.append(wts)
        gaussian_wts = torch.cat(gaussian_wts, dim=0 )
        gaussian_wts = F.normalize(gaussian_wts, dim=1, p=1) # NrxNb 
        
        #
        W_classifier = torch.cat((self.cls_score.weight, self.cls_score.bias.unsqueeze(1)),1).detach()  # Cx1025
        
       #%% 1 
        # #em = torch.mm(self.co_mats, W_classifier) (CxC) X (Cx1025)
        # em = torch.matmul(self.co_mats, W_classifier)  #10xCx1025
        # #em = torch.sum(em, dim=0)  # weighted sum of all comats 
        # em_fc = self.graph_fc(em) # CX 512  --> 10xCx512
        # em_fc = F.relu(em_fc)    # 10xCx512
        
        # em_fc_per = em_fc.permute(2,0,1)   # 512x10xC
         
        ## Class_probs/ Soft-mapping/ NrxC
        # Ms =  nn.Softmax(1)(scores).detach()  # detach/no
        # Ms_per = Ms.permute(1,0)
        
        # em_fc_softmap = torch.matmul(em_fc_per, Ms_per)     # 512x10xNr
        # em_fc_softmap = em_fc_softmap.permute(2,1,0)
        # em_fc_softmap_wt = em_fc_softmap * gaussian_wts.unsqueeze(2)
        
        # enc_feat = torch.sum(em_fc_softmap_wt,dim=1)   
        # enc_feat = torch.cat((enc_feat, x), dim=1)
        
        #%% 2 
        em = torch.matmul(self.co_mats, W_classifier)  # Nb x C x D
        em_per = em.permute(2,0,1)   # D x Nb x C
        
        Ms =  nn.Softmax(1)(scores).detach()  # Nr x C
        Ms_per = Ms.permute(1,0)   # C x Nr
        
        em_softmap = torch.matmul(em_per, Ms_per) # D x Nb x Nr 
        em_softmap = em_softmap.permute(2,1,0)  # Nr x Nb x D
        em_softmap_wt = em_softmap * gaussian_wts.unsqueeze(2) # Nr x Nb x D
        
        em_softmap_wt = torch.sum(em_softmap_wt,dim=1) # Nr x D
        
        enc_feat = self.graph_fc(em_softmap_wt)
        enc_feat = F.relu(enc_feat)
        enc_feat = torch.cat((enc_feat, x), dim=1)
        
        #%%        
        scores_enc = self.cls_score_enc(enc_feat)   # 
        proposal_deltas_enc = self.bbox_pred_enc(enc_feat)
        
        return scores, proposal_deltas, scores_enc, proposal_deltas_enc    