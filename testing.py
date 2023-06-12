from training import configure_net
import torch
from sklearn import metrics
from skimage import measure
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch import cuda
from time import time
import importlib
from os import path
from matplotlib import pyplot as plt
import numpy as np
from datasets.dataset import imagenet_denorm
from custom_metrics import f1_score, jaccard_score, precision_score
import matplotlib.patches as mpatch


def test_net(config, dataset, fold_info, sampler=None):
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if cuda.is_available() and config.gpu else 'cpu')
    net = configure_net(config.net_config, dataset.classes())

    net.load_state_dict(fold_info['model_state_dict'])
    net.to(device=device)
    net.train(mode=False)

    data_loader = DataLoader(dataset=dataset,
                             sampler=sampler,
                             batch_size=1,
                             num_workers=config.num_threads)

    acc, jcc, pre, f1 = 0, [], [], []
    labels = np.arange(0, dataset.classes())
    conf = np.zeros((dataset.classes(), dataset.classes()))
    total_time = 0
    for idx, (image, label) in enumerate(dataset):
        if not config.validation_stats and not config.generate_images:
            break

        image = image.unsqueeze(dim=0).to(device)
        label = label.unsqueeze(dim=0).to(device)
        t0 = time()
        prediction = net(image)
        t1 = time()
        total_time += t1 - t0

        # prediction = torch.rand((1, np.random.randint(0, 25), 704, 1024))
        pred_label = prediction.argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint)
        true_label = label.argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint)
        person = np.array([0, 0, 0])
        if config.person_detect:
            person += person_detect(config, true_label, pred_label, dataset)

        if config.generate_images:
            true_mask, pred_mask = dataset.pred_to_color_mask(true_label, pred_label)
            image = transforms.Normalize(**imagenet_denorm)(image).squeeze().movedim(0, -1).detach().cpu().numpy()
            plot_and_save(image, pred_mask, true_mask, idx, config, (dataset.colors()/255, dataset.classnames()))

        if config.validation_stats:
            pred_label = pred_label.flatten()
            true_label = true_label.flatten()
            acc += metrics.accuracy_score(true_label, pred_label, normalize=True)
            jcc.append(jaccard_score(true_label, pred_label, dataset.classes()))
            pre.append(precision_score(true_label, pred_label, dataset.classes()))
            f1.append(f1_score(true_label, pred_label, dataset.classes()))
            conf += metrics.confusion_matrix(true_label, pred_label, labels=labels)

    fold_results = {}
    inference_time = total_time / data_loader.__len__()
    print(f'Avg. inference time: {inference_time} s')
    if config.validation_stats:
        num_samples = len(data_loader)
        acc /= num_samples
        jcc = np.nanmean(np.vstack(jcc), axis=0)
        pre = np.nanmean(np.vstack(pre), axis=0)
        f1 = np.nanmean(np.vstack(f1), axis=0)
        # confusion gets normalization later

        fold_results = {'confusion': conf, 'acc': acc, 'jcc': jcc, 'pre': pre, 'f1': f1}
        print(fold_results)

    if config.training_charts:
        plot_training_charts(config, device)

    if config.person_detect:
        text_file = open(path.join(config.test_path, 'person.txt'), "w")
        text_file.write(f'TP: {person[0]}, FP: {person[1]}, FN: {person[2]}\n')
        text_file.close()


    del net
    return fold_results


def plot_training_charts(config, device):
    data = torch.load(path.join(config.train_path, 'training_results'), map_location=device)
    epoch = data[config.fold]['epoch']
    tacc = np.asarray(data[config.fold]['acc'])
    tacc = tacc.reshape((epoch + 1, -1)).mean(axis=1)
    loss = np.asarray(data[config.fold]['loss']).clip(0, 2)
    loss = loss.reshape((epoch + 1, -1)).mean(axis=1)
    epochs = np.arange(0, epoch + 1)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title(f'{config.name}, fold {config.fold}')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('acc', color=color)
    ax1.plot(epochs, tacc, color=color, label='Train acc')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, loss, color=color, label='Train loss')
    ax2.tick_params(axis='y', labelcolor=color)
    if 'acc_val' in data[config.fold]:
        acc_val = data[config.fold]['acc_val']
        loss_val = np.clip(data[config.fold]['loss_val'], -np.inf, 2)
        x_points = np.arange(config.validation_epochs, len(acc_val)*config.validation_epochs+1, step=config.validation_epochs)
        ax1.scatter(x_points, acc_val, c='red', label='Val acc')
        ax2.scatter(x_points, loss_val, c='blue', label='Val loss')
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.savefig(path.join(config.test_path, f'{config.name}_chart_fold{config.fold}.jpg'))
    # plt.show()
    plt.close(fig)


def plot_and_save(image, predicted_label, im_label, idx, config, legend):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.3), squeeze=False)
    [axis.set_axis_off() for axis in ax.flatten()]
    fig.tight_layout()
    ax[0, 0].imshow(image), ax[0, 1].imshow(predicted_label), ax[0, 2].imshow(im_label)
    ax[0, 0].set_title(f'Original')
    ax[0, 1].set_title('Prediction')
    ax[0, 2].set_title(f'Ground Truth')
    gs = ax[1, 0].get_gridspec()
    for axis in ax[1, :]:
        axis.remove()
    axbig = fig.add_subplot(gs[1, :])
    axbig.set_axis_off()
    colors, names = legend
    patches = [mpatch.Patch(color=c, label=n) for c, n in zip(colors, names)]
    axbig.legend(handles=patches, ncol=9, loc='upper left')
    plt.subplots_adjust(top=0.89)
    # plt.show()
    plt.savefig(path.join(config.test_path, f'{config.fold}_{idx}.jpg'))
    plt.close(fig)


def person_detect(config, gt_label, pred_label, dataset):
    h, w = config.net_config['input_size']
    fh, fw = 4000/h, 6000/w
    annotfile = path.join(config.tugraz_root, 'training_set/gt/bounding_box/bounding_boxes/person/imgIdToBBoxArray.p')
    rois = np.load(annotfile, allow_pickle=True)
    id = dataset._currid
    framerois = rois[id]
    person = (pred_label == 7).astype(int)
    labeled = measure.label(person)
    pred_max = np.max(labeled)
    seen = set()
    false_negative = 0
    total = 0
    true_positive = 0
    for roi in framerois:
        total += 1
        roi[:, 0] = roi[:, 0] / fh
        roi[:, 1] = roi[:, 1] / fw
        roi = roi.astype(int)
        crop = labeled[roi[0, 1]:roi[1, 1], roi[0, 0]:roi[1, 0]]
        labs = np.unique(crop)
        if len(labs) > 1:
            seen.update(labs)
            true_positive += 1
        else:
            false_negative += 1

    false_positive = pred_max - len(seen)
    return np.array([true_positive, false_positive, false_negative])
