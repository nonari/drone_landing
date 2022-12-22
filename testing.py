from training import configure_net
import torch
from sklearn import metrics
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch import cuda
import importlib
from os import path
from matplotlib import pyplot as plt
import numpy as np
from dataloader import tugraz_color_keys, imagenet_denorm


def test_net(config, dataset, fold_info, sampler=None):
    device = torch.device('cuda' if cuda.is_available() and config.gpu else 'cpu')
    net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
    net = configure_net(net_config, dataset.classes())

    net.load_state_dict(fold_info['model_state_dict'])
    net.to(device=device)
    net.train(mode=False)

    data_loader = DataLoader(dataset=dataset,
                             sampler=sampler,
                             batch_size=1,
                             num_workers=config.num_threads)

    arr = np.zeros(dataset.classes())
    acc, jcc, pre, f1 = 0, arr, arr, arr
    labels = np.arange(0, dataset.classes())
    conf = np.zeros((dataset.classes(), dataset.classes()))
    for idx, (image, label) in enumerate(data_loader):
        image = image.to(device)
        label = label.to(device)
        prediction = net(image)
        # prediction = torch.rand((1, np.random.randint(0, 25), 704, 1024))
        pred_label = prediction.argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint)
        true_label = label.argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint)
        pred_mask = tugraz_color_keys[pred_label]
        true_mask = tugraz_color_keys[true_label]
        image = transforms.Normalize(**imagenet_denorm)(image).squeeze().movedim(0, -1).detach().cpu().numpy()
        plot_and_save(image, pred_mask, true_mask, idx, config)
        pred_label = pred_label.flatten()
        true_label = true_label.flatten()

        acc += metrics.accuracy_score(true_label, pred_label, normalize=True)
        jcc = jcc + metrics.jaccard_score(true_label, pred_label, labels=labels, average=None, zero_division=1.0)
        pre = pre + metrics.precision_score(true_label, pred_label, labels=labels, average='macro', zero_division=1.0)
        f1 = f1 + metrics.f1_score(true_label, pred_label, labels=labels, average='macro', zero_division=1.0)
        conf += metrics.confusion_matrix(true_label, pred_label, labels=labels)

    num_samples = len(data_loader)
    acc /= num_samples
    jcc /= num_samples
    pre /= num_samples
    f1 /= num_samples
    # confusion gets normalization later

    fold_results = {'confusion': conf, 'acc': acc, 'jcc': jcc, 'pre': pre, 'f1': f1}
    print(fold_results)

    data = torch.load(path.join(config.train_path, 'training_results.json'), map_location=device)
    epoch = config.max_epochs if 'epoch' not in data[0] else data[config.fold]['epoch']
    tacc = np.asarray(data[config.fold]['acc'])[:-1]
    tacc = tacc.reshape((-1, tacc.shape[0] // epoch)).mean(axis=1)
    loss = np.asarray(data[config.fold]['loss'])[:-1]
    loss = loss.reshape((-1, loss.shape[0] // epoch)).mean(axis=1)
    epochs = np.arange(0, epoch)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_title(f'{config.name}, fold {config.fold}')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('acc', color=color)
    ax1.plot(epochs, tacc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(path.join(config.test_path, f'{config.name}_chart_fold{config.fold}.jpg'))
    # plt.show()

    return fold_results


def plot_and_save(image, predicted_label, im_label, idx, config):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.1))
    [axis.set_axis_off() for axis in ax]
    fig.tight_layout()
    ax[0].imshow(image), ax[1].imshow(predicted_label), ax[2].imshow(im_label)
    ax[0].set_title(f'Original')
    ax[1].set_title('Prediction')
    ax[2].set_title(f'Ground Truth')
    plt.subplots_adjust(top=0.89)
    plt.savefig(path.join(config.test_path, f'{(config.fold+1)*idx}.jpg'))
    plt.cla(), plt.clf()
