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


@MODELS.register_module()
class BECLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")
        loss = criterion(pred, target.long())
        loss_valid = loss[target != self.ignore_index]
        loss_valid_mean = torch.mean(loss_valid)
        return loss_valid_mean


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
            B, H, W = probs.size()
            probs = probs.view(B, 1, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
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


def inverse_sigmoid(x, sign="A"):
    x = x.to(torch.float32)
    while x >= 1 - 1e-5:
        x = x - 1e-5

    while x < 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)


@MODELS.register_module()
class Geo_scal_loss(nn.Module):
    def __init__(self, ignore_index=255, free_index=0, loss_weight=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.free_index = free_index
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, ignore_index=None, free_index=None):
        if ignore_index is None:
            ignore_index = self.ignore_index
        if free_index is None:
            free_index = self.free_index

        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = target != 255
        nonempty_target = target != 0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )


@MODELS.register_module()
class Sem_scal_loss(nn.Module):
    def __init__(self, ignore_index=255, loss_weight=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, ignore_index=None):
        # Get softmax probabilities
        if ignore_index is None:
            ignore_index = self.ignore_index
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)
        loss = 0
        count = 0
        mask = target != 255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):
            # Get probability of class i
            p = pred[:, i, :, :, :]

            # Remove unknown voxels
            target_ori = target
            p = p[mask]
            t = target[mask]

            completion_target = torch.ones_like(t)
            completion_target[t != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target))
                    loss_specificity = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count
