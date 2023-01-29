CONFIG = {
    'net': 'pspnet',
    'input_size': (704, 1024),
    'encoder': 'inceptionv4',
    'pretrained': True,
    'loss': 'nn.BCEWithLogitsLoss',
    'optimizer': {
        'name': 'optim.RMSprop',
        'params': {
            'lr': 1e-4,
            'momentum': 0.9,
            'weight_decay': 1e-5
          }
    },
    'lr_scheduler': {
        'name': 'optim.lr_scheduler.CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 10,
            'eta_min': 1e-5,
            'verbose': True
          }
    }
}