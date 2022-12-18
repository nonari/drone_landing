from os import makedirs, path
from shutil import rmtree
from config import Config
from training import train_net
import fire
from torch.utils.data import SubsetRandomSampler
from dataloader import TUGrazDataset
from glob import glob
from os import path
import torch
from sklearn.model_selection import KFold
import random
from glob import glob
from testing import test_net


def last_checkpoint(config):
    paths = glob(config.checkpoint_path + '/*')
    paths = sorted(paths, key=lambda p: (path.basename(p).split('_')[0], path.basename(p).split('_')[1]))
    if len(paths) == 0:
        raise Exception(f'No checkpoint found at {config.checkpoint_path}, check path or disable -resume')

    return paths[-1]


def folds_strategy(config):
    idx_seed = random.randint(0, 9999)
    config.fold = 0
    checkpoint = None

    dataset = TUGrazDataset(config)
    if config.resume:
        checkpoint_path = last_checkpoint(config)
        checkpoint = torch.load(checkpoint_path)
        config.fold = checkpoint['fold']
        idx_seed = checkpoint['idx_seed']

    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=idx_seed)
    folds = list(kfold.split(dataset))

    for fold, (train_idx, _) in enumerate(folds[config.fold:]):
        sampler = SubsetRandomSampler(train_idx)
        config.fold = fold
        train_net(config, dataset, idx_seed, sampler=sampler, checkpoint=checkpoint)
        checkpoint = None


def train(**kwargs):
    opt = Config()

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # create directories
    if not opt.resume:
        if path.exists(opt.save_path):
            if opt.override:
                rmtree(opt.save_path)
            else:
                r = input(f'Will delete {opt.save_path}. Proceed? y/n')
                if r == 'y':
                    rmtree(opt.save_path)
                else:
                    print('!!!! Execution aborted !!!!')
                    exit(0)

    makedirs(opt.save_path, exist_ok=True)
    makedirs(opt.model_path, exist_ok=True)
    makedirs(opt.checkpoint_path, exist_ok=True)
    makedirs(opt.test_path, exist_ok=True)
    makedirs(opt.train_path, exist_ok=True)

    if opt.folds > 1:
        folds_strategy(opt)


def test(**kwargs):
    opt = Config()

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.folds > 1:
        folds_test()


def folds_test(config):
    dataset = TUGrazDataset(config)
    model_paths = glob(path.join(config.model_path, '*'))
    model_paths = sorted(model_paths, key=lambda p: int(path.basename(p)))

    seed_info = torch.load(model_paths[0])

    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=seed_info['idx_seed'])
    folds = list(kfold.split(dataset))

    for fold, (_, test_idx) in enumerate(folds):
        sampler = SubsetRandomSampler(test_idx)
        fold_info = torch.load(model_paths[fold])
        test_net(config, dataset, fold_info, sampler=sampler)


if __name__ == '__main__':
    fire.Fire()
