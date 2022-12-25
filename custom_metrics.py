import numpy as np
from sklearn import metrics


def calc_acc(output, target):
    predictions = output.argmax(dim=1).squeeze()
    target = target.argmax(dim=1).squeeze()
    correct = (predictions == target).sum().item()
    accuracy = correct / target.size().numel()

    return accuracy


def precision_score(true, pred, num_classes):
    label_classes = set(np.unique(pred))
    all_classes = set(np.arange(num_classes))
    not_predicted = all_classes - label_classes
    result = metrics.precision_score(true, pred, labels=np.arange(num_classes), average=None, zero_division=0)
    result[list(not_predicted)] = np.nan
    return result


def jaccard_score(true, pred, num_classes):
    pred_classes = set(np.unique(true))
    all_classes = set(np.arange(num_classes))
    not_predicted = all_classes - pred_classes
    result = metrics.jaccard_score(true, pred, labels=np.arange(num_classes), average=None, zero_division=0)
    result[list(not_predicted)] = np.nan
    return result


def f1_score(true, pred, num_classes):
    true_pred_classes = set(np.unique(true))
    true_pred_classes.update(np.unique(pred))
    not_present = set(np.arange(num_classes)) - true_pred_classes
    result = metrics.f1_score(true, pred, labels=np.arange(num_classes), average=None, zero_division=0)
    result[list(not_present)] = np.nan
    return result
