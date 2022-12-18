from training import configure_net
import torch
from torch.utils.data.dataloader import DataLoader
from torch import cuda
import importlib
from os import path


def test_net(config, dataset, fold_info, sampler=None):
    device = torch.device('cuda' if cuda.is_available() and config.gpu else 'cpu')
    net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
    net = configure_net(net_config, dataset.classes())

    results = torch.load(path.join(config.train_path, 'training_results.json'))
    net.load_state_dict(fold_info['model_state_dict'])
    net.to(device=device)
    net.train(mode=False)

    data_loader = DataLoader(dataset=dataset,
                             sampler=sampler,
                             batch_size=1,
                             num_workers=config.num_threads)

    for image, label in data_loader:
        prediction = net(image)
