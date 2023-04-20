CONFIG = {
    'net': {
        'name': 'safeuav',
        'params': {
            'last': 'sigmoid',
            'init_nb': 24
        }
    },
    'input_size': (704, 1024),
    'loss': {
        'name': 'custom_models.losses.BCEDiceLoss',
        'params': {}
    },
    'optimizer': {
        'name': 'torch.optim.RMSprop',
        'params': {
        }
    },
    'lr_scheduler': {
        'name': 'mock_scheduler.MockScheduler',
        'params': {}
    }
}
