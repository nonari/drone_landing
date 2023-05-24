import torch
from torch import nn, Tensor
from torch.nn import functional as F


def dice_coeff(y_pred, y_true):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    score = (2. * intersection + smooth) / (union + smooth)
    return score


def dice_coeff_multiclass(y_pred, y_true):
    smooth = 1.
    pred_soft = y_pred.softmax(dim=1)
    intersection = torch.sum(y_true * pred_soft, dim=(0, 2, 3))
    union = torch.sum(y_true + pred_soft, dim=(0, 2, 3))

    score = torch.mean((2. * intersection + smooth) / (union + smooth))
    return score


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_pred, y_true):
        loss = self.bce_loss(y_pred, y_true) + self.dice_loss(y_pred, y_true)

        return loss


class BCEDiceAvgLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceAvgLoss()

    def forward(self, y_pred, y_true):
        loss = self.bce_loss(y_pred, y_true) + self.dice_loss(y_pred, y_true)

        return loss


class DiceAvgLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        clipped = torch.clip(dice_coeff_multiclass(y_pred, y_true), 0, 1)
        return -torch.log((clipped + 1) / 2)


class DiceAvgLossOld(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return 1 - dice_coeff_multiclass(y_pred, y_true)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return 1 - dice_coeff(y_pred, y_true)


class CEWeightDiceAvgLoss(nn.Module):
    def __init__(self, w, config):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
        self.dice_loss = DiceAvgLoss()

    def forward(self, y_true, y_pred):
        loss = self.ce_loss(y_true, y_pred) + self.dice_loss(y_true, y_pred)

        return loss


class CEWeightDiceLoss(nn.Module):
    def __init__(self, w, config):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        loss = self.ce_loss(y_true, y_pred) + self.dice_loss(y_true, y_pred)

        return loss


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_true, y_pred):
        loss = self.ce_loss(y_true, y_pred)

        return loss


class BCELL(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcell = nn.BCEWithLogitsLoss()

    def forward(self, y_true, y_pred):
        loss = self.bcell(y_true, y_pred)

        return loss


class BCELLW(nn.Module):
    def __init__(self, w, config):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
        self.bcell = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w).to(device).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3))

    def forward(self, y_true, y_pred):
        loss = self.bcell(y_true, y_pred)

        return loss


class BCELLWDiceAvg(nn.Module):
    def __init__(self, w, config):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
        self.bcell = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w).to(device).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3))
        self.diceavg = DiceAvgLoss()

    def forward(self, y_true, y_pred):
        loss = self.bcell(y_true, y_pred) + self.diceavg(y_true, y_pred)

        return loss


class BCELLDiceAvg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcell = nn.BCEWithLogitsLoss()
        self.diceavg = DiceAvgLoss()

    def forward(self, y_true, y_pred):
        loss = self.bcell(y_true, y_pred) + self.diceavg(y_true, y_pred)

        return loss


class BCELLDiceAvgOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcell = nn.BCEWithLogitsLoss()
        self.diceavg = DiceAvgLossOld()

    def forward(self, y_true, y_pred):
        loss = self.bcell(y_true, y_pred) + self.diceavg(y_true, y_pred)

        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        target = torch.argmax(target, dim=1)
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


class FocalLoss(nn.Module):
    def __init__(self,
                 config=None,
                 alpha=None,
                 gamma: float = 0.5,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        super().__init__()
        if config is not None:
            device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
            alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        y = torch.argmax(y, dim=1)
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class FocalDiceAvg(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.focal_loss = FocalLoss(**kwargs)
        self.dice_loss = DiceAvgLoss()

    def forward(self, pred, target):
        fl = self.focal_loss(pred, target)
        dl = self.dice_loss(pred, target)
        return fl + 1.5*dl
