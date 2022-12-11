from config import Config
from training import train_net
import fire
from dataloader import TUGrazDataset


def train(**kwargs):
    """ Get options """
    opt = Config()

    print('we are here you not')
    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.folds > 1:
        if opt.resume:
            pass
    else:
        dataset = TUGrazDataset(opt, shuffle=True)
        train_net(opt, dataset)


if __name__ == '__main__':
    fire.Fire()

