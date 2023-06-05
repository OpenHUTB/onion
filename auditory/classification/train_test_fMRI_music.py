'''
train_test.py
利用和fMRI相同的训练集和测试集进行模型的训练和测试
Test Accuracy: 56.04%

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''

import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)

import os
from time import *
import pickle

from torch.optim.lr_scheduler import ReduceLROnPlateau
import data_manager
from src.CNN2D.model.custom_upchannel import *  # 默认
from src.CNN2D.model.custom_RCNN import *       # 加入GRU
from hparams import hparams


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams):
        # self.model =upchannel(hparams)  # 默认2D卷积
        self.model = RCNN(hparams)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate, momentum=hparams.momentum, weight_decay=1e-6, nesterov=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
        self.learning_rate = hparams.learning_rate
        self.stopping_rate = hparams.stopping_rate
        self.device = torch.device("cpu")

        if hparams.device > 0:
            torch.cuda.set_device(hparams.device - 1)
            self.model.cuda(hparams.device - 1)
            self.criterion.cuda(hparams.device - 1)
            self.device = torch.device("cuda:" + str(hparams.device - 1))

    # Accuracy function works like loss function in PyTorch
    def accuracy(self, source, target):
        source = source.max(1)[1].long().cpu()
        target = target.long().cpu()
        correct = (source == target).sum().item()

        return correct/float(source.size(0))

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        epoch_acc = 0
        for batch, (x, y) in enumerate(dataloader):  # 总共 480*128*128 （测试集10类，每类48个）
            x = x.to(self.device)  # bs*128*128
            y = y.to(self.device)  # bs*1 (类别为0-9)

            prediction = self.model(x)
            loss = self.criterion(prediction, y.long())
            acc = self.accuracy(prediction, y.long())

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()
            epoch_acc += prediction.size(0)*acc

        epoch_loss = epoch_loss/len(dataloader.dataset)
        epoch_acc = epoch_acc/len(dataloader.dataset)

        return epoch_loss, epoch_acc

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate

        return stop


def device_name(device):
    if device == 0:
        device_name = 'CPU'
    else:
        device_name = 'GPU:' + str(device - 1)

    return device_name


# Test Accuracy: 96.16%
# Run time： 33.33 s
def test_model():
    # runner = Runner(hparams)
    with open(os.path.join(hparams.data_path, 'speedup', 'runner.pkl'), 'rb') as f:
        runner = pickle.load(f)
        runner = runner[0]

    # test_loader: x(2216*128*128), y(2216)
    # len(test_loader)=18
    with open(os.path.join(hparams.data_path, 'speedup', 'data_loader.pkl'), 'rb') as f:
        train_loader, valid_loader, test_loader = pickle.load(f)

    # 用于保存需要查看的网络层特征
    _model_feats = []
    def _store_feats(layer, input, output):
        _model_feats.append(output.clone().detach().cpu().numpy())
    runner.model._classifier._modules['0'].register_forward_hook(_store_feats)  # 注册保存中间层特征的钩子

    test_loss, test_acc = runner.run(test_loader, 'eval')  # 进行模型的前向预测

    print(len(_model_feats))      # 18
    print(_model_feats[0].shape)  # 128*512
    np.save(os.path.join(hparams.data_path, 'speedup', '_classifier_0_feats.npy') , _model_feats)
    # np.load(os.path.join(hparams.data_path, 'speedup', '_classifier_0_feats.npy'))

    print("Testing Finished")
    print("Test Accuracy: %.2f%%" % (100 * test_acc))


# 训练和测试一起
def main():
    save_model_dir = os.path.join(hparams.data_path, 'model')
    if os.path.exists(save_model_dir) is False:
        os.mkdir(save_model_dir)
    speedup_dir = os.path.join(hparams.data_path, 'speedup')
    if os.path.exists(speedup_dir) is False:
        os.mkdir(speedup_dir)
    speedup_loader_path = os.path.join(speedup_dir, 'data_loader.pkl')
    if os.path.exists(speedup_loader_path):
        # Getting back the objects:
        with open(os.path.join(hparams.data_path, 'speedup', 'data_loader.pkl'), 'rb') as f:
            train_loader, valid_loader, test_loader = pickle.load(f)
    else:
        train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)  # 841 s
        # 保存准备好的数据，给下一次运行使用:
        with open(speedup_loader_path, 'wb') as f:
            # 在3.8 版更改: 默认协议版本是4，解决保存的文件大于 4G 的问题
            pickle.dump([train_loader, valid_loader, test_loader], f,
                        protocol=4)  # protocol=4 解决 OverflowError: cannot serialize a bytes object larger than 4 G

    runner = Runner(hparams)

    max_valid_acc = 0  # 随便设置一个比较小的数
    print('Training on ' + device_name(hparams.device))
    for epoch in range(hparams.num_epochs):
        train_loss, train_acc = runner.run(train_loader, 'train')
        valid_loss, valid_acc = runner.run(valid_loader, 'eval')

        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
              (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc))

        # 保存验证集上精度最高的模型
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            print("save model")
            torch.save(runner.model.state_dict(), os.path.join(save_model_dir + str(round(valid_acc, 4)) + '.pth'))

        if runner.early_stop(valid_loss, epoch + 1):
            break

    test_loss, test_acc = runner.run(test_loader, 'eval')
    print("Training Finished")
    print("Test Accuracy: %.2f%%" % (100*test_acc))

    runner_path = os.path.join(hparams.data_path, 'speedup', 'runner.pkl')
    with open(runner_path, 'wb') as f:
        # 每次都是覆盖上次运行的过程
        pickle.dump([runner], f, protocol=4)


if __name__ == '__main__':
    begin_time = time()

    # main()

    # 增强8倍后的测试集（音频顺序记录在 `run_dataset/test_order_in_DNN.txt` 中），
    # 激活保存在 `speedup/_classifier_0_feats.npy` 中：
    # (240*8)*128*128 -> 15*128*512 （15个batch, 每个batch 128 个音频，每个音频的特征长度为 512）
    test_model()

    end_time = time()
    run_time = end_time - begin_time
    print('Run time： %.2f s' % run_time)
