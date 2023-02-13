CONFIG = {
    'net': 'safeuav',
    'input_size': (704, 1024),
    'loss': 'custom_models.losses.BCEDiceLoss',
    'optimizer': {
        'name': 'optim.RMSprop',
        'params': {
            'lr': 1e-4,
            'alpha': 0.9,
            'eps': 1e-7
          }
    },
}