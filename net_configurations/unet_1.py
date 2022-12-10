CONFIG = {
    'net': "unet",
    'encoder': "resnet34",
    'pretrained': True,
    'loss': 'nn.BCEWithLogitsLoss',
    'optimizer': {
        'name': 'optim.SGD',
        'params': {
            'lr': 1e-4,
            'nesterov': True,
            'momentum': 0.9,
            'weight_decay': 1e-4
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