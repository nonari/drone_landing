from os import makedirs, path


class Config(object):
    def __init__(self):
        # basic options, project name should be same as dataset directory name
        self.name = 'UNet'
        self.dataset_name = 'TU_Graz'
        self.comment = 'Basic test'

        self.train = True
        self.tugraz_root = ''

        self.save_path = path.join('./checkpoints', self.name)
        self.model_path = path.join(self.save_path, 'models')
        self.checkpoint_path = path.join(self.save_path, 'checkpoints')
        self.test_path = path.join(self.save_path, 'test_results')

        # transforms
        self.resize = True
        self.flip = True
        self.num_threads = 4

        # network
        self.model = 'unet'
        self.model_config = None
        self.gpu = True                      # use GPU?

        # save
        self.save_every = 10  # epochs

        # create directories
        makedirs(self.save_path, exist_ok=True)
        makedirs(self.model_path, exist_ok=True)
        makedirs(self.checkpoint_path, exist_ok=True)
        makedirs(self.test_path, exist_ok=True)