from config import Config
from training import train_net
import fire
from dataloader import TUGrazDataset
from glob import glob
from os import path
import torch


def last_checkpoint(config):
    paths = glob(config.checkpoint_path + '/*')
    paths = sorted(paths, key=lambda p: (path.basename(p).split('_')[0], path.basename(p).split('_')[1]))
    return paths[-1]


def train(**kwargs):
    opt = Config()

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    idx = None
    opt.fold = 0
    checkpoint = None

    if opt.resume:
        checkpoint_path = last_checkpoint(opt)
        checkpoint = torch.load(checkpoint_path)
        opt.fold = checkpoint['fold']
        idx = checkpoint['idx']

    dataset = TUGrazDataset(opt, folds=opt.folds, shuffle=True, idx_ord=idx)
    for f in range(opt.fold, opt.folds):
        opt.fold = f
        dataset.change_fold(f)
        train_net(opt, dataset, checkpoint=checkpoint)
        checkpoint = None


if __name__ == '__main__':
    fire.Fire()

