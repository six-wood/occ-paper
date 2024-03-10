import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.models.losses.lovasz_loss import lovasz_hinge, lovasz_softmax_flat
from torch.cuda.amp import autocast

# Copyright (c) OpenMMLab. All rights reserved.
"""Directly borrowed from mmsegmentation.

Modified from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytor
ch/lovasz_losses.py Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim
Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from mmengine.utils import is_list_of

from mmdet3d.registry import MODELS


def flatten_probs(probs: torch.Tensor, labels: torch.Tensor, ignore_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions and labels in the batch. Remove tensors whose labels
    equal to 'ignore_index'.

    Args:
        probs (torch.Tensor): Predictions to be modified.
        labels (torch.Tensor): Labels to be modified.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to None.

    Return:
        tuple(torch.Tensor, torch.Tensor): Modified predictions and labels.
    """
    if probs.dim() != 2:  # for input with P*C
        if probs.dim() == 3:
            # assumes output of a sigmoid layer
            B, C, N = probs.size()
            probs = probs.permute(0, 2, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
        if probs.dim() == 4:
            # assumes output of a softmax layer
            B, C, H, W = probs.size()
            probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
        if probs.dim() == 5:
            # assumes output of a softmax layer
            B, C, H, W, D = probs.size()
            probs = probs.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_softmax(
    probs: torch.Tensor,
    labels: torch.Tensor,
    classes: Union[str, List[int]] = "present",
    per_sample: bool = False,
    class_weight: List[float] = None,
    reduction: str = "mean",
    avg_factor: Optional[int] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): Class probabilities at each
            prediction (between 0 and 1) with shape [B, C, H, W].
        labels (torch.Tensor): Ground truth labels (between 0 and
            C - 1) with shape [B, H, W].
        classes (Union[str, list[int]]): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Defaults to 'present'.
        per_sample (bool): If per_sample is True, compute the loss per
            sample instead of per batch. Defaults to False.
        class_weight (list[float], optional): The weight for each class.
            Defaults to None.
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_sample is True. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_sample is True.
            Defaults to None.
        ignore_index (Union[int, None]): The label index to be ignored.
            Defaults to 255.

    Returns:
        torch.Tensor: The calculated loss.
    """

    if per_sample:
        loss = [
            lovasz_softmax_flat(*flatten_probs(prob.unsqueeze(0), label.unsqueeze(0), ignore_index), classes=classes, class_weight=class_weight)
            for prob, label in zip(probs, labels)
        ]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(*flatten_probs(probs, labels, ignore_index), classes=classes, class_weight=class_weight)
    return loss


@MODELS.register_module()
class OccLovaszLoss(nn.Module):
    """LovaszLoss.

    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.

    Args:
        loss_type (str): Binary or multi-class loss.
            Defaults to 'multi_class'. Options are "binary" and "multi_class".
        classes (Union[str, list[int]]): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Defaults to 'present'.
        per_sample (bool): If per_sample is True, compute the loss per
            sample instead of per batch. Defaults to False.
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_sample is True. Defaults to 'mean'.
        class_weight ([list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_type: str = "multi_class",
        classes: Union[str, List[int]] = "present",
        per_sample: bool = False,
        reduction: str = "mean",
        class_weight: Optional[List[float]] = None,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        assert loss_type in (
            "binary",
            "multi_class",
        ), "loss_type should be \
                                                    'binary' or 'multi_class'."

        if loss_type == "binary":
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax
        assert classes in ("all", "present") or is_list_of(classes, int)
        if not per_sample:
            assert (
                reduction == "none"
            ), "reduction should be 'none' when \
                                                        per_sample is False."

        self.classes = classes
        self.per_sample = per_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self, cls_score: torch.Tensor, label: torch.Tensor, avg_factor: int = None, reduction_override: str = None, **kwargs) -> torch.Tensor:
        """Forward function."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # if multi-class loss, transform logits to probs
        if self.cls_criterion == lovasz_softmax:
            cls_score = F.softmax(cls_score, dim=1)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, self.classes, self.per_sample, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs
        )
        return loss_cls


def one_hot(label: torch.Tensor, n_classes: int, requires_grad: bool = True) -> torch.Tensor:
    """Return One Hot Label."""
    device = label.device
    one_hot_label = torch.eye(n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


@MODELS.register_module()
class BoundLoss(nn.Module):
    """Boundary loss."""

    def __init__(self, theta0=3, theta=5, loss_weight: float = 1.0) -> None:
        super(BoundLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            pred (Tensor): The output from model.
            gt (Tensor): Ground truth map.

        Returns:
            Tensor: Loss tensor.
        """
        pred = F.softmax(pred, dim=1)
        n, c, _, _ = pred.shape

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return self.loss_weight * loss
