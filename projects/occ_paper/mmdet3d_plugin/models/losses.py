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
        empty_probs = pred[:, free_index]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = target != ignore_index
        nonempty_target = target != free_index
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        eps = 1e-5
        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / (nonempty_probs.sum() + eps)
        recall = intersection / (nonempty_target.sum() + eps)
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum() + eps)
        with autocast(False):
            return (
                F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, "A"), torch.ones_like(precision))
                + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, "B"), torch.ones_like(recall))
                + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, "C"), torch.ones_like(spec))
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

        # with autocast(False):
        pred = F.softmax(pred, dim=1)
        loss = 0
        count = 0
        mask = target != ignore_index
        n_classes = pred.shape[1]
        for i in range(0, n_classes):
            # Get probability of class i
            p = pred[:, i]

            # Remove unknown voxels
            p = p[mask]
            t = target[mask]

            completion_target = torch.ones_like(t)
            completion_target[t != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p) + 1e-5)
                    # print("precision", precision)
                    loss_precision = F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, "D"), torch.ones_like(precision))
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) + 1e-5)
                    # loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    # print("recall", recall)
                    loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, "E"), torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-5)
                    # print("specificity", specificity)
                    loss_specificity = F.binary_cross_entropy_with_logits(inverse_sigmoid(specificity, "F"), torch.ones_like(specificity))
                    loss_class += loss_specificity
                loss += loss_class
                # print(i, loss_class, loss_recall, loss_specificity)
        l = loss / count
        if torch.isnan(l):
            from IPython import embed

            embed()
            exit()
        return l

def one_hot(label: Tensor,
            n_classes: int,
            requires_grad: bool = True) -> Tensor:
    """Return One Hot Label."""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


@MODELS.register_module()
class BoundaryLoss(nn.Module):
    """Boundary loss."""

    def __init__(self, theta0=3, theta=5, loss_weight: float = 1.0) -> None:
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
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
        gt_b = F.max_pool2d(
            1 - one_hot_gt,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2)
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
