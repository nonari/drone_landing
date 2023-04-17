CONFIG = {
    'net': {
        'name': 'safeuav',
        'params': {}
    },
    'input_size': (704, 1024),
    'loss': {
        'name': 'custom_models.losses.BCEDiceAvgLoss',
        'params': {}
    },
    'optimizer': {
        'name': 'torch.optim.RMSprop',
        'params': {}
    },
}