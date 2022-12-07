from config import Config
import fire


def train(**kwargs):
    """ Get options """
    opt = Config()

    print('we are here you not')
    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)


if __name__ == '__main__':
    fire.Fire()

