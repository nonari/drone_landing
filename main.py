from os import makedirs, path
from shutil import rmtree
from config import Config, TestConfig
from training import train_net
import fire
from torch.utils.data import SubsetRandomSampler
from dataloader import TUGrazDataset
from aeroscapes import AeroscapesDataset
from os import path
import torch
from sklearn.model_selection import KFold
import random
from glob import glob
from testing import test_net
import numpy as np


def select_dataset(config):
    dataset_name = config.dataset_name
    if dataset_name == 'TU_Graz':
        dataset = TUGrazDataset(config)
    elif dataset_name == 'aeroscapes':
        dataset = AeroscapesDataset(config)
    else:
        raise Exception(f'Dataset name {dataset_name}, not found.')

    return dataset


def save_execution_data(config):
    data = {'max_epochs': config.max_epochs,
            'batch_size': config.batch_size,
            'folds': config.folds,
            'model_config': config.model_config,
            'idx_seed': config.idx_seed,
            'dataset_name': config.dataset_name}

    data_path = path.join(config.train_path, 'execution_info')
    torch.save(data, data_path)


def load_execution_data(config):
    exec_info_path = path.join(config.train_path, 'execution_info')
    if not path.exists(exec_info_path):
        raise Exception(f'Execution data not found at {config.train_path}, check path or disable -resume')
    info = torch.load(exec_info_path)
    config.max_epochs = max(info['max_epochs'], config.max_epochs)
    config.batch_size = info['batch_size']
    config.folds = info['folds']
    config.model_config = info['model_config']
    config.idx_seed = info['idx_seed']
    config.dataset_name = info['dataset_name']

    train_info_path = path.join(config.train_path, 'training_results')
    config._training_status = torch.load(train_info_path)


def last_executions(config):
    executed_folds = config._training_status.keys()

    if len(executed_folds) == 0:
        return [], 0

    executed_folds = sorted(executed_folds, key=lambda x: int(x))

    paths = []
    fst_fold = -1
    for fold in executed_folds:
        curr_epoch = config._training_status[fold]['epoch']
        if curr_epoch < config.max_epochs - 1:
            if fst_fold < 0:
                fst_fold = fold
            paths.append(path.join(config.checkpoint_path, f'{fold}_{curr_epoch}'))

    if len(paths) == 0:
        raise Exception('Execution is already complete, you might want to increase epochs')

    return paths


def folds_strategy(config):
    idx_seed = random.randint(0, 9999) if config.idx_seed is None else config.idx_seed
    config.fold = 0

    dataset = TUGrazDataset(config)
    checkpoint_paths = []
    if config.resume:
        checkpoint_paths, config.fold = last_executions(config)

    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=idx_seed)
    folds = list(kfold.split(dataset))

    for fold, (train_idx, _) in list(enumerate(folds))[config.fold:]:
        checkpoint = None
        if len(checkpoint_paths) > 0:
            checkpoint = torch.load(checkpoint_paths.pop(0))
        config.fold = fold
        sampler = SubsetRandomSampler(train_idx)
        train_net(config, dataset, idx_seed, sampler=sampler, checkpoint=checkpoint)


def train(**kwargs):
    name = kwargs['name']
    opt = Config(name=name)

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # create directories
    if opt.resume:
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
    name = kwargs['name']
    opt = TestConfig(name=name)

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    used_dataset = torch.load(path.join(opt.train_path, 'execution_info'))['dataset_name']
    curr_dataset = opt.dataset_name

    if used_dataset != curr_dataset:
        test_only_one(opt)

    if opt.folds > 1:
        folds_test(opt)


def test_only_one(config):
    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
    dataset = select_dataset(config)
    model_path = path.join(config.model_path, f'{config.model}')
    fold_info = torch.load(model_path, map_location=device)
    result = test_net(config, dataset, fold_info)


def folds_test(config):
    dataset = TUGrazDataset(config)
    model_paths = glob(path.join(config.model_path, '*'))
    model_paths = sorted(model_paths, key=lambda p: int(path.basename(p)))

    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
    idx_seed = torch.load(path.join(config.train_path, 'execution_info'))['idx_seed']

    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=idx_seed)
    folds = list(kfold.split(dataset))

    results = []
    for fold, (_, test_idx) in enumerate(folds):
        config.fold = fold
        sampler = SubsetRandomSampler(test_idx)
        fold_info = torch.load(model_paths[fold], map_location=device)
        result = test_net(config, dataset, fold_info, sampler=sampler)
        results.append(result)

    if config.validation_stats:
        summary = summarize_results(results)
        torch.save(summary, path.join(config.test_path, 'metrics_summary'))


def summarize_results(results):
    acc = [r['acc'] for r in results]
    jcc = [r['acc'] for r in results]
    pre = [r['acc'] for r in results]
    f1 = [r['acc'] for r in results]
    conf = [r['acc'] for r in results]

    acc = np.asarray(acc)
    jcc = np.nanmean(np.vstack(jcc), axis=0)
    pre = np.nanmean(np.vstack(pre), axis=0)
    f1 = np.nanmean(np.vstack(f1), axis=0)

    conf = np.dstack(conf)
    conf = conf / conf.astype(np.float).sum(axis=1)

    final_results = {'confusion': conf, 'acc': acc, 'jcc': jcc, 'pre': pre, 'f1': f1}
    return final_results


if __name__ == '__main__':
    fire.Fire()
