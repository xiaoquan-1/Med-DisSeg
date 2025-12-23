import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve

def Object(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2.0 * x / (x ** 2 + 1 + sigma_x + np.finfo(np.float64).eps)

    return score

def S_Object(pred, gt):
    pred_fg = pred.copy()
    pred_fg[gt != 1] = 0.0
    O_fg = Object(pred_fg, gt)

    pred_bg = (1 - pred.copy())
    pred_bg[gt == 1] = 0.0
    O_bg = Object(pred_bg, 1 - gt)

    u = np.mean(gt)
    Q = u * O_fg + (1 - u) * O_bg

    return Q


def centroid(gt):
    if np.sum(gt) == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2

    else:
        x, y = np.where(gt == 1)
        return int(np.mean(x).round()), int(np.mean(y).round())


def divide(gt, x, y):
    LT = gt[:x, :y]
    RT = gt[x:, :y]
    LB = gt[:x, y:]
    RB = gt[x:, y:]

    w1 = LT.size / gt.size
    w2 = RT.size / gt.size
    w3 = LB.size / gt.size
    w4 = RB.size / gt.size

    return LT, RT, LB, RB, w1, w2, w3, w4


def ssim(pred, gt):
    x = np.mean(pred)
    y = np.mean(gt)
    N = pred.size

    sigma_x2 = np.sum((pred - x) ** 2 / (N - 1 + np.finfo(np.float64).eps))
    sigma_y2 = np.sum((gt - y) ** 2 / (N - 1 + np.finfo(np.float64).eps))

    sigma_xy = np.sum((pred - x) * (gt - y) / (N - 1 + np.finfo(np.float64).eps))

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(np.float64).eps)
    elif alpha == 0 and beta == 0:
        Q = 1
    else:
        Q = 0

    return Q


def S_Region(pred, gt):
    x, y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divide(gt, x, y)
    pred1, pred2, pred3, pred4, _, _, _, _ = divide(pred, x, y)

    Q1 = ssim(pred1, gt1)
    Q2 = ssim(pred2, gt2)
    Q3 = ssim(pred3, gt3)
    Q4 = ssim(pred4, gt4)

    Q = Q1 * w1 + Q2 * w2 + Q3 * w3 + Q4 * w4

    return Q
def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def AlignmentTerm(pred, gt):
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)

    align_pred = pred - mu_pred
    align_gt = gt - mu_gt

    align_mat = 2 * (align_gt * align_pred) / (align_gt ** 2 + align_pred ** 2 + np.finfo(np.float64).eps)

    return align_mat


def EnhancedAlighmentTerm(align_mat):
    enhanced = ((align_mat + 1) ** 2) / 4
    return enhanced

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #不再做sigmoid
        #inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #不再做sigmoid
        #inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class MultiClassBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        loss = []
        for i in range(inputs.shape[1]):
            yp = inputs[:, i]
            yt = targets[:, i]
            BCE = F.binary_cross_entropy(yp, yt, reduction='mean')

            if i == 0:
                loss = BCE
            else:
                loss += BCE

        return loss
#为了训练synapse用的Dice多分类
class SoftDiceMultiClassLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, ignore_index=None):
        super().__init__()
        self.C, self.smooth, self.ignore_index = num_classes, smooth, ignore_index
    def forward(self, logits, target):  # logits:(B,C,H,W), target:(B,H,W) long
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target.clamp_min(0), num_classes=self.C).permute(0,3,1,2).float()
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            probs, onehot = probs*mask, onehot*mask
        dims=(0,2,3)
        inter=(probs*onehot).sum(dims)
        denom=probs.sum(dims)+onehot.sum(dims)
        dice=(2*inter+self.smooth)/(denom+self.smooth)
        return 1-dice.mean()
#为了训练synapse用的CE+Dice
import torch
import torch.nn as nn
import torch.nn.functional as F

class CEDiceLoss(nn.Module):
    """
    多类 CE + 前景平均 SoftDice
    - logits: 任意常见形状，内部会规范到 (N,C,H,W)
    - target: 类别 id，内部会规范到 (N,H,W)
    - 可选类别权重 ce_weight（Tensor[C]），会注册为 buffer，自动随设备迁移
    """
    def __init__(self, num_classes, ce_weight=None, ignore_index=None,
                 smooth=1.0, ce_coeff=0.5, dice_coeff=0.5):
        super().__init__()
        self.C = num_classes
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.ce_coeff = float(ce_coeff)
        self.dice_coeff = float(dice_coeff)

        # 处理并注册 CE 权重
        if ce_weight is not None and not torch.is_tensor(ce_weight):
            ce_weight = torch.tensor(ce_weight, dtype=torch.float32)
        if ce_weight is not None:
            self.register_buffer("ce_weight", ce_weight.float())
            weight_arg = self.ce_weight
        else:
            self.ce_weight = None
            weight_arg = None

        self.ce = nn.CrossEntropyLoss(
            weight=weight_arg,
            ignore_index=ignore_index if ignore_index is not None else -100
        )

    def _normalize_logits_target(self, logits, target):
        """把 logits 规范到 (N,C,H,W)，target 规范到 (N,H,W)"""
        C = self.C

        # ---- logits -> (N,C,H,W)
        if logits.dim() == 3:
            # (C,H,W) 或 (H,W,C)
            if logits.size(0) == C:
                logits = logits.unsqueeze(0)                        # -> (1,C,H,W)
            elif logits.size(-1) == C:
                logits = logits.permute(2, 0, 1).unsqueeze(0)       # (H,W,C)->(C,H,W)->(1,C,H,W)
            else:
                raise ValueError(f"logits 3D but cannot infer channel C={C}, got {tuple(logits.shape)}")
        elif logits.dim() == 4:
            # (N,C,H,W) 或 (N,H,W,C)
            if logits.size(1) != C and logits.size(-1) == C:
                logits = logits.permute(0, 3, 1, 2)                 # NHWC -> NCHW
            if logits.size(1) != C:
                raise ValueError(f"logits channel != C ({logits.size(1)} vs {C}), shape={tuple(logits.shape)}")
        else:
            raise ValueError(f"Unsupported logits dim={logits.dim()}, shape={tuple(logits.shape)}")

        # ---- target -> (N,H,W)
        if target.dim() == 2:
            target = target.unsqueeze(0)                             # (H,W)->(1,H,W)
        elif target.dim() == 3:
            # 通常已经是 (N,H,W)
            pass
        elif target.dim() == 4:
            # (N,1,H,W) 或 (N,C,H,W onehot)
            if target.size(1) == 1:
                target = target[:, 0, :, :]                          # -> (N,H,W)
            elif target.size(1) == C:
                target = target.argmax(dim=1)                        # onehot -> (N,H,W)
            else:
                raise ValueError(f"Unsupported target shape={tuple(target.shape)} for C={C}")
        else:
            raise ValueError(f"Unsupported target dim={target.dim()}, shape={tuple(target.shape)}")

        return logits, target

    def softdice_fg_mean(self, logits, target):
    # logits: (B,C,H,W), target: (B,H,W)
        probs = F.softmax(logits, dim=1)                       # (B,C,H,W)
        onehot = F.one_hot(target.clamp_min(0), self.C).permute(0,3,1,2).float()  # (B,C,H,W)

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)             # (B,1,H,W)
            probs  = probs  * mask
            onehot = onehot * mask

        # 这里是关键：对 3D 张量 (B,H,W) 求和 -> 维度应为 (0,1,2)
        sum_dims = (0, 1, 2)

        dice_per_class = []
        for c in range(1, self.C):  # 跳过背景 0
            pc = probs[:, c]        # (B,H,W)
            oc = onehot[:, c]       # (B,H,W)
            inter = (pc * oc).sum(dim=sum_dims)
            denom = pc.sum(dim=sum_dims) + oc.sum(dim=sum_dims)
            dice_c = (2 * inter + self.smooth) / (denom + self.smooth)
            dice_per_class.append(dice_c.mean())

        if len(dice_per_class) == 0:
            return torch.tensor(0.0, device=logits.device)

        return 1 - torch.stack(dice_per_class).mean()


    def forward(self, logits, target):
        # 交叉熵期望 (N,C,H,W) & (N,H,W)
        logits, target = self._normalize_logits_target(logits, target)

        # 确保 CE 的权重 tensor 在同一设备
        if self.ce_weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce_weight.to(logits.device)

        loss_ce   = self.ce(logits, target)               # scalar
        loss_dice = self.softdice_fg_mean(logits, target) # scalar
        return self.ce_coeff * loss_ce + self.dice_coeff * loss_dice



""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def mae(y_true, y_pred):
    sum=0
    for i in range(len(y_true)):
        sum+=abs(y_true[i]-y_pred[i])
    return sum/len(y_true)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

