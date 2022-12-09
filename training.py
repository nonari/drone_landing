import importlib


g = importlib.import_module('model_configurations.unet_1')
print(g.CONFIG)
