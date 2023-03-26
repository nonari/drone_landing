from os import path


class Config(object):
    def __init__(self, name):
        # basic options, project name should be same as dataset directory name
        self.name = name
        self.dataset_name = 'TU_Graz'
        self.comment = 'Basic test'

        self.train = True

        self.rural_root = '/home/nonari/Documentos/ruralscapes'

        self.aeroscapes_root = '/home/nonari/Documentos/aeroscapes'

        self.tugraz_root = '/home/nonari/Documentos/semantic_drone_dataset_semantics_v1.1/semantic_drone_dataset/'
        self.tugraz_images_loc = 'low_res_images'
        self.tugraz_labels_loc = 'low_res_label_images'

        self.save_path = path.join('./executions', self.name)
        self.model_path = path.join(self.save_path, 'models')
        self.checkpoint_path = path.join(self.save_path, 'checkpoints')
        self.test_path = path.join(self.save_path, 'test_results')
        self.train_path = path.join(self.save_path, 'training_info')

        self.num_threads = 2

        # do not change
        self.fold = 0

        # network
        self.model_config = 'unet_resnet34'
        self.gpu = True


class TrainConfig(Config):
    def __init__(self, name):
        super().__init__(name)
        self._training = False

        # transforms
        self.augment = True

        # checkpoint frequency
        self.save_every = 50  # epochs

        # continue or expand last execution
        self.resume = False

        # wipe execution dir
        self.override = False

        # this is override on resume
        self.idx_seed = 42
        self.batch_size = 1
        self.folds = 5
        self.max_epochs = 100
        # for validation
        self.validation_epochs = 10
        self.stop_after_miss = 2
        self.delta = 0.01


class TestConfig(Config):
    def __init__(self, name):
        super().__init__(name)
        self._training = False
        self.model = 0
        self.generate_images = True
        self.training_charts = True
        self.validation_stats = True
