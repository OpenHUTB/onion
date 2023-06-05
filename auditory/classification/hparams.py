import argparse
import os
import platform


class HParams(object):
    def __init__(self):
        self.is_debug = False
        plat = platform.system().lower()
        if plat == 'windows':
            self.data_path = 'D:\\dong\\data\\brain\\auditory'
        elif plat == 'linux':
            self.data_path = '/data3/dong/data/brain/auditory'  # 听觉处理的主工作目录

        # self.dataset_path = os.path.join(self.data_path, 'genres') # 原始版本
        self.dataset_path = os.path.join(self.data_path, 'run_dataset')  # 对核磁共振的刺激进行分类
        self.run_dataset_path = os.path.join(self.data_path, 'run_dataset')  # 存放对应fMRI的音频文件（用于分类的）

        # '/data3/dong/data/brain/auditory/feature_augment'  # 产生的增强数据保存目录
        # self.feature_path = os.path.join(self.data_path, 'feature_augment')  # 原始版本
        self.feature_path = os.path.join(self.data_path, 'feature_augment_fMRI')    # 对核磁共振的刺激进行分类
        self.feature_fMRI_path = os.path.join(self.data_path, 'feature_augment_fMRI')

        # 在原始版本的基础之上增加rock和blues
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

        # Feature Parameters
        self.sample_rate = 22050
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 512
        self.num_mels = 128
        self.feature_length = 1024

        # Training Parameters
        self.device = 1  # 0: CPU, 1: GPU0, 2: GPU1, ...
        self.batch_size = 128
        if self.is_debug is True:
            self.num_epochs = 1
        else:
            self.num_epochs = 26
        self.learning_rate = 1e-2
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-6
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 5
        self.workers = 16

    # Function for pasing argument and set hParams
    def parse_argument(self, print_argument=True):
        parser = argparse.ArgumentParser()
        for var in vars(self):
            value = getattr(hparams, var)
            argument = '--' + var
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args()
        for var in vars(self):
            setattr(hparams, var, getattr(args, var))

        if print_argument:
            print('----------------------')
            print('Hyper Paarameter Settings')
            print('----------------------')
            for var in vars(self):
                value = getattr(hparams, var)
                print(var + ":" + str(value))
            print('----------------------')


hparams = HParams()
hparams.parse_argument()
