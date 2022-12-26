from matplotlib import pyplot as plt
import seaborn as sea


def confusion(conf, class_names, config):
    fig = plt.figure(figsize=(16, 14))
    ax = plt.subplot()
    sea.heatmap(conf, annot=True, ax=ax, fmt='g')  # annot=True to annotate cells

    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize=10)
    plt.yticks(rotation=0)

    plt.title('Refined Confusion Matrix', fontsize=20)

    plt.savefig(f'{config.name}_conf.png')
    plt.show()
