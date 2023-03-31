CONFIG = {
    'net': {
        'name': 'safeuav',
        'params': {
            'last': 'sigmoid'
        }
    },
    'input_size': (704, 1024),
    'loss': {
        'name': 'custom_models.losses.BCEDiceLoss',
        'params': {}
    },
    'optimizer': {
        'name': 'optim.RMSprop',
        'params': {
            'lr': 1e-4,
            'alpha': 0.9,
            'eps': 1e-7
          }
    },
}