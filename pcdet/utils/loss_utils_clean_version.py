import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import box_utils
import math


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


def sigmoid_focal_cls_loss(input, target, gamma=2.0, alpha=0.25):
    pred_sigmoid = torch.sigmoid(input)
    alpha_weight = target * alpha + (1 - target) * (1 - alpha)
    pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
    focal_weight = alpha_weight * torch.pow(pt, gamma)

    bce_loss = SigmoidFocalClassificationLoss.sigmoid_cross_entropy_with_logits(input, target)

    loss = focal_weight * bce_loss
    return loss


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0
        assert pred.size() == target.size()
        loss = torch.abs(pred - target)
        return loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights

        return pos_loss + neg_loss



class ShiftDisLoss(nn.Module):
    def __init__(self):
        super(ShiftDisLoss, self).__init__()

    def point_to_line_v2(self, points, lines):
        x0, y0 = points[:, 0], points[:, 1] 
        x1, y1 = lines[:, 0, 0], lines[:, 0, 1]
        x2, y2 = lines[:, 1, 0], lines[:, 1, 1]
        numerator = ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        squared_numerator = numerator**2
        denominator = (y2-y1)**2 + (x2-x1)**2
        return squared_numerator / denominator

    def forward(self, pred_bbox3d, gt_bbox3d, weights=None):
        loss = 0
        for k in range(pred_bbox3d.shape[0]): 
            reg_weight = weights[k] if weights is not None else None
            sums = sum(reg_weight!=0)

            pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d[k].clone()) 
            gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d[k].clone())
            pred_x1, pred_y1 = pred_box_corners[:, 0, 0], pred_box_corners[:, 0, 1]
            pred_x2, pred_y2 = pred_box_corners[:, 1, 0], pred_box_corners[:, 1, 1]
            pred_x3, pred_y3 = pred_box_corners[:, 2, 0], pred_box_corners[:, 2, 1]
            pred_x4, pred_y4 = pred_box_corners[:, 3, 0], pred_box_corners[:, 3, 1]
            gt_x1, gt_y1 = gt_box_corners[:, 0, 0], gt_box_corners[:, 0, 1]
            gt_x2, gt_y2 = gt_box_corners[:, 1, 0], gt_box_corners[:, 1, 1]
            gt_x3, gt_y3 = gt_box_corners[:, 2, 0], gt_box_corners[:, 2, 1]
            gt_x4, gt_y4 = gt_box_corners[:, 3, 0], gt_box_corners[:, 3, 1]
            
            corners_pred = torch.cat((pred_x1.unsqueeze(1), pred_y1.unsqueeze(1), pred_x2.unsqueeze(1), pred_y2.unsqueeze(1), pred_x3.unsqueeze(1), pred_y3.unsqueeze(1), pred_x4.unsqueeze(1), pred_y4.unsqueeze(1)), dim=1).reshape(-1, 4, 2) # [num_bboxes, 4, 2]
            corners_gt = torch.cat((gt_x1.unsqueeze(1), gt_y1.unsqueeze(1), gt_x2.unsqueeze(1), gt_y2.unsqueeze(1), gt_x3.unsqueeze(1), gt_y3.unsqueeze(1), gt_x4.unsqueeze(1), gt_y4.unsqueeze(1)), dim=1).reshape(-1, 4, 2)

            # sort the corners in the following order:
            #     """
            #     3 -------- 1            1 -------- 3
            #     |          |            |          |
                
            #     |          |            |          |
            #     2 -------- 0            0 -------- 2
                            
            #                 *(0, 0)
            #         ...                    ...
            #
            #     """

            squared_pred = torch.sum(corners_pred ** 2, dim=2)
            squared_gt = torch.sum(corners_gt ** 2, dim=2)
            sorted_pred_ind = torch.argsort(squared_pred, dim=1)
            sorted_gt_ind = torch.argsort(squared_gt, dim=1)
            corners_pred = torch.gather(corners_pred, 1, sorted_pred_ind.unsqueeze(-1).expand(-1, -1, 2))
            corners_gt = torch.gather(corners_gt, 1, sorted_gt_ind.unsqueeze(-1).expand(-1, -1, 2))

            squared_pred_2 = corners_pred[:, 1:3, 0] ** 2
            squared_gt_2 = corners_gt[:, 1:3, 0] ** 2
            sorted_pred_ind_2 = torch.argsort(squared_pred_2, dim=1)
            sorted_gt_ind_2 = torch.argsort(squared_gt_2, dim=1)
            corners_pred_2 = torch.gather(corners_pred[:, 1:3].clone(), 1, sorted_pred_ind_2.unsqueeze(-1).expand(-1, -1, 2)) # [num_bboxes, 2, 2]
            corners_gt_2 = torch.gather(corners_gt[:, 1:3].clone(), 1, sorted_gt_ind_2.unsqueeze(-1).expand(-1, -1, 2))

            corners_pred[:, 1] = corners_pred_2[:, 0]
            corners_pred[:, 2] = corners_pred_2[:, 1]
            corners_gt[:, 1] = corners_gt_2[:, 0]
            corners_gt[:, 2] = corners_gt_2[:, 1]

            diag = torch.norm(corners_gt[:, 1] - corners_gt[:, 2], dim=1)**2

            point_to_line_2a = self.point_to_line_v2(corners_pred[:, 1], corners_gt[:, [0,1]]) 
            point_to_line_2b = self.point_to_line_v2(corners_pred[:, 2], corners_gt[:, [0,2]])
            point_to_line_2c = self.point_to_line_v2(corners_pred[:, 0], corners_gt[:, [0,3]])
            point_to_line_2 = (point_to_line_2a + point_to_line_2b + point_to_line_2c)/ diag 

            assert point_to_line_2.shape == reg_weight.shape
            loss += torch.sum(point_to_line_2 * reg_weight)

        loss_total = loss / pred_bbox3d.shape[0] 
        return loss_total



