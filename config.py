from os import makedirs, path


class Config(object):
    def __init__(self):
        # basic options, project name should be same as dataset directory name
        self.name = 'UNet_test1'
        self.dataset_name = 'TU_Graz'
        self.comment = 'Basic test'

        self.train = True
        self.tugraz_root = '/semantic_drone_dataset_semantics_v1.1/semantic_drone_dataset/'
        self.tugraz_images_loc = 'low_res_images'
        self.tugraz_labels_loc = 'low_res_label_images'

        self.save_path = path.join('./executions', self.name)
        self.model_path = path.join(self.save_path, 'models')
        self.checkpoint_path = path.join(self.save_path, 'checkpoints')
        self.test_path = path.join(self.save_path, 'test_results')
        self.train_path = path.join(self.save_path, 'training_results')

        # transforms
        self.resize = True
        self.flip = True
        self.num_threads = 2

        # network
        self.model_config = 'unet_1'
        self.gpu = True                  # use GPU?
        self.classes = None

        # save
        self.save_every = 2  # epochs

        # training
        self.batch_size = 1
        self.resume = True
        self.folds = 4
        self.max_epochs = 4

        # create directories
        makedirs(self.save_path, exist_ok=True)
        makedirs(self.model_path, exist_ok=True)
        makedirs(self.checkpoint_path, exist_ok=True)
        makedirs(self.test_path, exist_ok=True)
        makedirs(self.train_path, exist_ok=True)
