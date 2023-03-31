CONFIG = {
    'net': {
        'name': 'unet',
        'params': {
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet'
        }
    },
    'input_size': (704, 1024),
    'loss': 'nn.BCEWithLogitsLoss',
    'optimizer': {
        'name': 'optim.RMSprop',
        'params': {
            'lr': 1e-2,
            'momentum': 0,
            'weight_decay': 0
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