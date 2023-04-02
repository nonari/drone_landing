CONFIG = {
    'net': {
        'name': 'pspnet',
        'params': {
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet'
        }
    },
    'input_size': (704, 1024),
    'encoder': 'resnet34',
    'pretrained': True,
    'loss': 'torch.nn.BCEWithLogitsLoss',
    'optimizer': {
        'name': 'torch.optim.RMSprop',
        'params': {
            'lr': 1e-4,
            'momentum': 0.9,
            'weight_decay': 1e-5
          }
    },
    'lr_scheduler': {
        'name': 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 10,
            'eta_min': 1e-5,
            'verbose': True
          }
    }
}
