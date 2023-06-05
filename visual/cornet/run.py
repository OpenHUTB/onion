import os
import argparse
import time
import glob
import pickle
import subprocess
import shlex
import io
import pprint
import importlib

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

import cornet

from PIL import Image
Image.warnings.simplefilter('ignore')

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--model', choices=['Z', 'R', 'S'], default='Z',
                    help='which model to train, Z(smallest), R(Recurrent), S(reSnet)')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')


FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)


def get_model(pretrained=False):
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    model = getattr(cornet, f'cornet_{FLAGS.model.lower()}')
    if FLAGS.model.lower() == 'r':
        model = model(pretrained=pretrained, map_location=map_location, times=FLAGS.times)
    else:
        model = model(pretrained=pretrained, map_location=map_location)

    if FLAGS.ngpus == 0:
        model = model.module  # remove DataParallel
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.1,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=5,  # how often save model weigths
          save_model_secs=60*10  # how often save model (in sec)
          ):

    model = get_model()
    trainer = ImageNetTrain(model)
    validator = ImageNetVal(model)

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    trainer.model.train()

            if FLAGS.output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(FLAGS.output_path + 'results.pkl', 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, FLAGS.output_path +
                                   'latest_checkpoint.pth.tar')
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, FLAGS.output_path +
                                   f'epoch_{epoch:02d}.pth.tar')

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()


def test(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    适合测试小图片集合，如果有成千上万的图片或者需要长时间来提取特征，考虑使用 `torchvision.datasets.ImageFolder`
    使用 `ImageNetVal` 作为一个例子

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    model = get_model(pretrained=True)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize,imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        _model_feats.append(np.reshape(output, (len(output), -1)).numpy())

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    # 检查是否可以使用 gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model_feats = []
        fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.*')))
        if len(fnames) == 0:
            raise f'No files found in {FLAGS.data_path}'
        for fname in tqdm.tqdm(fnames):
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise f'Unable to load {fname}'
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            # im = im.to(device)  # to gpu
            model(im)
            model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)

    if FLAGS.output_path is not None:
        fname = f'CORnet-{FLAGS.model}_{layer}_{sublayer}_feats.npy'
        np.save(os.path.join(FLAGS.output_path, fname), model_feats)


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if FLAGS.ngpus > 0:
            target = target.cuda(non_blocking=True)
        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        record['learning_rate'] = self.lr.get_lr()[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'val_in_folders'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


def score_model(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    给模型进行类脑相似性打分
    适合测试小图片集合，如果有成千上万的图片或者需要长时间来提取特征，考虑使用 `torchvision.datasets.ImageFolder`
    使用 `ImageNetVal` 作为一个例子

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    model = get_model(pretrained=True)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize,imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])
    model.eval()

    # 对模型进行类脑相似性打分
    import functools
    from model_tools.activations.pytorch import load_preprocess_images
    from model_tools.activations.pytorch import PytorchWrapper

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing)

    from brainscore import score_model
    from model_tools.brain_transformation import ModelCommitment

    model = ModelCommitment(identifier='my-model', activations_model=activations_model,
                            # 指定需要考虑的层
                            layers=['V4.conv1', 'V4.conv2', 'V4.conv3'])
    score = score_model(model_identifier=model.identifier, model=model,
                        benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
    print(score)


# 获得指定层的激活
# todo: gpu 加速
def get_activations(stimulus_set, layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    适合测试小图片集合，如果有成千上万的图片或者需要长时间来提取特征，考虑使用 `torchvision.datasets.ImageFolder`
    使用 `ImageNetVal` 作为一个例子

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    model = get_model(pretrained=True)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize,imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        _model_feats.append(np.reshape(output, (len(output), -1)).numpy())

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    # 检查是否可以使用 gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model_feats = []
        image_ids = []
        object_names = []
        # image_path = stimulus_set.get_image(stimulus_set['image_id'][0])
        # fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.*')))
        image_path = stimulus_set.get_image(stimulus_set['image_id'][0])
        image_dir = os.path.split(image_path)[0]
        fnames = stimulus_set['filename']
        if len(fnames) == 0:
            raise f'No files found in {FLAGS.data_path}'
        # for fname in tqdm.tqdm(fnames):  # 进度条
        for i in range(len(stimulus_set)):
            fname = stimulus_set['image_file_name'][i]
            fname = os.path.join(image_dir, fname)
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise f'Unable to load {fname}'
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            # im = im.to(device)  # to gpu
            model(im)
            model_feats.append(_model_feats[time_step])
            image_ids.append(stimulus_set['image_id'][i])
            object_names.append(stimulus_set['object_name'][i])
            # 显示进度：
            if i % 50 == 0:
                print(i)
        model_feats = np.concatenate(model_feats)
        # image_ids = np.concatenate(image_ids)
        # object_names = np.concatenate(object_names)

    # 根据获取的深度网络的特征构建用于计算类脑相似性的数据集 ModelFeaturesAssembly
    from brainio.assemblies import ModelFeaturesAssembly
    prediction = ModelFeaturesAssembly(model_feats,
                                       coords={'image_id': ('presentation', image_ids),
                                               'object_name': ('presentation', object_names),
                                               'neuroid_id': ('neuroid', np.arange(512)),  # 神经元的个数512应该和皮层记录的位置数维度一致？
                                               'region': ('neuroid', [0] * 512)},
                                       dims=['presentation', 'neuroid'])  # 图像为presentation展示，神经元的激活为 neuroid
    return prediction


# score --model S --data_path /data3/dong/data/brain/visual/cornet/image_dicarlo_hvm-public/ --output_path /data3/dong/data/brain/visual/cornet/output
def score(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
    # （1）在两个系统之间使用线性回归，来进行线性映射（比如从神经激活到神经放电率）；
    # 偏最小二乘回归（英语：Partial Least Squares regression， PLS回归）
    # 采用对变量X和Y都进行分解的方法，从变量X和Y中同时提取成分(通常称为因子)，再将因子按照它们之间的相关性从大到小排列。
    regression = pls_regression()  # 1: 定义回归
    # （2）它计算在保留（hold-out）图像上所预测的放电率之间的相关性；
    correlation = pearsonr_correlation()  # 2: 定义相关性
    # （3）并将所有这些都包含在交叉验证中以估计泛化能力。
    metric = CrossRegressedCorrelation(regression, correlation)  # 3: 包含在交叉验证中

    # 刺激对应的皮层激活集 NeuronRecordingAssembly： 3200*168（图片数目 * 皮层记录的位置数目）
    from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark
    benchmark = MajajHongITPublicBenchmark()
    benchmark_assembly = benchmark._assembly

    # 图片刺激集 3200 张图片，位于 `/home/d/.brainio/image_dicarlo_hvm-public/`
    import brainscore
    neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")  # 256*148480*1
    stimulus_set = neural_data.attrs['stimulus_set']
    # image_path = stimulus_set.get_image(stimulus_set['image_id'][0])
    prediction = get_activations(stimulus_set)  # layer='decoder', sublayer='avgpool' 3200*512
    # model_feats = np.load("/data3/dong/data/brain/visual/cornet/output/CORnet-S_decoder_avgpool_feats.npy")  # for debug

    brain_like_score = metric(source=prediction, target=benchmark_assembly)  # AssertionError
    print(brain_like_score)  # array([0.513728, 0.004886])


if __name__ == '__main__':
    # fire几乎可以不改变原始代码就可以生成命令行接口CLIs(Command Line Interfaces)
    # 第一个参数就是所调用的函数名
    fire.Fire(command=FIRE_FLAGS)

'''
test --model S --data_path <path to your image folder> 
-o <path to save features> --ngpus 1q

python run.py test --model S --data_path /data2/whd/workspace/sot/data -o /data2/whd/workspace/sot/model

python run.py train --model S --workers 20 --ngpus 1 --step_size 20 --epochs 43 --lr .1 --data_path /data2/whd/workspace/data/Imagenet2012
'''