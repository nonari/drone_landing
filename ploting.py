from matplotlib import pyplot as plt
import seaborn as sea
import torch
from os import path
from dataloader import tugraz_classnames
from aeroscapes import aeroscapes_classnames


def confusion(conf, class_names, test_path):
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

    plt.savefig(path.join(test_path, 'confusion.jpg'))
    plt.show()


if __name__ == '__main__':
    d = torch.load('./executions/PSPNet_r32/test_results_alt/metrics_summary', map_location='cpu')
    confusion(d['confusion'], aeroscapes_classnames, './executions/PSPNet_r32/test_results_alt')
