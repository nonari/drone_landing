from torch import optim


class MockScheduler:
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def get_last_lr(self):
        pass

    def get_lr(self):
        pass

    def print_lr(self, is_verbose, group, lr, epoch=None):
        pass

    def step(self, epoch=None):
        pass
