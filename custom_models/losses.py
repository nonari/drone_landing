import torch
from torch import nn


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    score = (2. * intersection + smooth) / (union + smooth)
    return score


def dice_coeff_multiclass(y_true, y_pred):
    smooth = 0.1
    intersection = torch.sum(y_true * y_pred, dim=(0, 2, 3))
    union = torch.sum(y_true + y_pred, dim=(0, 2, 3))

    score = torch.mean((2. * intersection + smooth) / (union + smooth))
    return score


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        loss = self.bce_loss(y_true, y_pred) + self.dice_loss(y_true, y_pred)

        return loss


class BCEDiceAvgLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceAvgLoss()

    def forward(self, y_true, y_pred):
        loss = self.bce_loss(y_true, y_pred) + self.dice_loss(y_true, y_pred)

        return loss


class DiceAvgLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return 1 - dice_coeff_multiclass(y_true, y_pred)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return 1 - dice_coeff(y_true, y_pred)


class CEWeightDiceLoss(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.bce_loss = nn.CrossEntropyLoss(weight=w)
        self.dice_loss = DiceAvgLoss()

    def forward(self, y_true, y_pred):
        loss = self.bce_loss(y_true, y_pred) + self.dice_loss(y_true, y_pred)

        return loss


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_true, y_pred):
        loss = self.ce_loss(y_true, y_pred)

        return loss
