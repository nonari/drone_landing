import torch
from torch import nn


def dice_coeff(y_pred, y_true):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    score = (2. * intersection + smooth) / (union + smooth)
    return score


def dice_coeff_multiclass(y_pred, y_true):
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