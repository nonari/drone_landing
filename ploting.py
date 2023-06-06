from matplotlib import pyplot as plt
import seaborn as sea
import torch
from datasets.aeroscapes import aeroscapes_classnames
from sklearn.metrics import ConfusionMatrixDisplay


def confusion_old(conf, class_names, location):
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=class_names)
    conf_disp.plot(xticks_rotation=25, values_format='.2f')
    # plt.show()
    plt.savefig(location, bbox_inches="tight")


def confusion(conf, class_names, location):
    fig = plt.figure(figsize=(16, 14))
    ax = plt.subplot()
    sea.heatmap(conf, annot=False, ax=ax, fmt='g')  # annot=True to annotate cells

    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize=10)
    plt.yticks(rotation=0)

    plt.title('Refined Confusion Matrix', fontsize=20)

    plt.savefig(location)


if __name__ == '__main__':
    d = torch.load('./executions/PSPNet_r32/test_results_alt/metrics_summary', map_location='cpu')
    confusion(d['confusion'], aeroscapes_classnames, './executions/PSPNet_r32/test_results_alt')
