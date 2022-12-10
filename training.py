import importlib
import segmentation_models_pytorch as smp


def configure_net(config):
    model_config = importlib.import_module(f'model_configurations.{config.model_config}').CONFIG
    net_type = model_config['net']
    if net_type == "unet":
        net = smp.Unet(
            encoder_name=model_config['encoder'],
            encoder_weights='imagenet' if model_config['pretrained'] else None,
            in_channels=3,
            classes=config.classes
        )
    else:
        raise NotImplementedError

    return net


def train(config):
    net = configure_net(config)


