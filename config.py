from os import path


class Config(object):
    def __init__(self, name):
        # basic options, project name should be same as dataset directory name
        self.name = name
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
        self.model_config = 'unet_resnet34'
        self.gpu = True                  # use GPU?
        self.classes = None

        # save
        self.save_every = 50  # epochs

        # training
        self.batch_size = 1
        self.resume = False
        self.override = False
        self.folds = 5
        self.max_epochs = 100
