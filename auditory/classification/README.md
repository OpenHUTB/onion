# 音乐风格分类（Pytorch）

We discuss the application of convolutional neural networks for the task of music genre classification. 
We focus in the small data-set case and how to build CNN architecture. We start with data augmentation strategy in music domain, and compare well-known architecture in the 1D, 2D, sample CNN with law data and augmented-data.
Moreover, we suggest best performance CNN architecture in small-data music-genre classification. Then, we compare normalization method and optimizers. we will be discussed to see how to obtain the model that fits better in the music genre classification. 
Finally, we evaluate its performance in the GTZAN dataset, used in a lot of works, in order to compare the performance of our approach with the state of the art.

[源地址](https://github.com/dutlzn/MUSIC-SRC/tree/5494a8bf60eaa8040800e79eb4b47e3b2c5d2a4c/Music_Genre_Classification_Pytorch-master/Music_Genre_Classification_Pytorch-master) 已经没了。

## 代码适配到 fMRI 刺激分类
修改了超参数配置文件，并修改了 `data_manager.py` 的
```
def get_label(file_name, hparams):
    genre = genre.split('_')[-1]  # added for fMRI music
```

原模型测试精度只有 Test Accuracy: 58.54%

恢复，删除一个全连接层：Test Accuracy: 54.79%
删除一个卷积层：Test Accuracy: 50.21%


## 应用
搜索、推荐


## TODO
* 参考CorNet实现A1->belt->PB->T2/T3实现分类网络

* 删除原来genres目录并重新解压

* 保存测试集上效果最好的模型
* 合并数据增强和特征抽取


### Dataset
We use the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset which has been the most widely used in the music genre classification task. 
The dataset contains 30-second audio files including 10 different genres including reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. 
使用全部风格的音乐。
For this homework, we are going to use a subset of GTZAN with only 8 genres. You can download the subset from [this link](https://drive.google.com/file/d/1rHw-1NR_Taoz6kTfJ4MPR5YTxyoCed1W/view?usp=sharing).

Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  

### Data augmentation
Data augmentation is the process by which we create new synthetic training samples by adding small perturbations on our initial training set. 
The objective is to make our model invariant to those perturbations and enhance its ability to generalize. In order to this to work adding the perturbations must conserve the same label as the original training sample.
- Add Noise
- Shift
- Speed Change
- Pitch Shift
- Pitch and Speed
- Multiply Value
- Percussive

<img src="/img/augmentation.png">
<img src="/img/mel.png">

# 小波散射
(学习视频)[https://ww2.mathworks.cn/videos/signal-classification-using-wavelet-scattering-1603097304980.html]

(代码)[https://github.com/mathworks/deep-learning-for-time-series-data]


# 运行
## 环境安装
* Python 3.7 (recommended)
* Numpy
* Librosa
* PyTorch 1.0

### 问题
module 'librosa' has no attribute 'output'
```shell script
pip install librosa==0.7.2
```
No module named 'numba.decorators' 
```shell script
pip install numba==0.48.0
```

## 数据说明

`test_list.txt` 记录了 `40行*6列` （`ExpValOrder.mat`）行的测试音频。

`train_list.txt` 记录了 `4800` （`ExpTrnOrder.mat` 中的 40 行* 12 列 * 10,增强十次）行的训练音频。

利用fMRI构建用于音频分类的文件名 `run_dataset/reggae/Test_run-06-40_reggae.00013_15s.wav` 中 `06` 表示位于 `ExpValOrder.mat` 中的列号（从1开始），`40` 表示位于 `ExpValOrder.mat` 中的行号（从1开始）。


## 数据增强
运行数据增加脚本之前重新解压/data3/dong/data/brain/auditory/genres.tar.gz，并拷贝classification/dataset/gtzan的文件到data/brain/auditory/genres目录下
```py
audio_augmentation.py
```

只对训练数据进行增强。


## 抽取特诊

使用mel-spectogram进行特征的抽取：
```py
feature_extraction.py
```

## 训练和测试

检测和修改hparams.py中的超参数，并运行：
```py
train_test.py
```



# 结果
自己按照7:1.5:1.5分割数据集（使用GRU）：
```shell script
Train Acc: 0.9991
Valid Acc: 0.9480
Test Accuracy: 96.16%
Run time： 1730.51 s
```

DataLoader中的workers数量由1增加到10
```shell script
Run time of this training and testing： 1694
```

DataLoader中的workers数量由1增加到16
```shell script
Run time of this training and testing： 1676
```

使用初始代码：
```cmd output
Train Acc: 0.9979
Valid Acc: 0.7650
Test Accuracy: 81.22%
```
初始代码加入GRU：
```cmd output
Train Acc: 0.9999
Valid Acc: 0.7558
Test Accuracy: 81.99%
Run time: 820s
```

The model with the best validation accuracy is the 4Layer CNN with 77%. The test accuracy of this model is 83.39%. 
Sample_rate 22050 used in feature engineering, fft size 1024, win size 1024, hop size 512, num mels 128, feature length 1024. 
We also recorded 26 epochs based on early stop criteria. Stochastic gradient descent was used, and learning rate 0.01, momentum 0.9, weight decay 1e-6, using nesterov showed the best performance.

<img src="/img/cnn.png">

Model | Train Acc | Valid Acc  | Train Acc(Augmented) | Valid Acc(Augmented) | Test Acc
:---:|:---:|:---:|:---:|:---:|:---:
5L-1D CNN | 0.97 | 0.55 | 0.99 | 0.70
AlexNet | 0.98 | 0.63 | 0.99 | 0.72 
VGG11 | 0.99 | 0.68 | 0.99 | 0.76
VGG13 | 0.97 | 0.68 | 0.99 | 0.74 
VGG16 | 0.99 | 0.69 | 0.99 | 0.75 
VGG19 | 0.98 | 0.67 | 0.99 | 0.74 
GooLeNet | 0.75 | 0.57 | 0.99 | 0.65 
ResNet34 | 0.99 | 0.63 | 0.99 | 0.70 
ResNet50 | 0.99 | 0.61 | 0.99 | 0.69 
DenseNet | 0.98 | 0.66 | 0.99 | 0.76
Sample CNN Basic Block | 0.13 | 0.13 | 0.15 | 0.13 
4L-2D CNN | 0.93 | 0.62 | 0.95 | 0.77 | 83.39
4L-2D CNN + GRU | 0.92 | 0.64 | 0.99 | 0.76 | 81.55

# 实验
## 基础
使用自定义的二维卷积和GRU作为修改的初始版本，根据听觉皮层处理音乐的通路进行改进。

## 其他自带模型
* [x] [Custom 1D CNN]() (5Layer 1D-CNN)
* [x] [Alexnet]() (Alexnet)
* [x] [Very Deep Convolutional Networks for Large-Scale Image Recognition]()(Vgg16)
  + https://arxiv.org/abs/1409.1556
* [x] [Goin deeper with convolutions]() (Inception)
  + https://arxiv.org/abs/1409.4842
* [x] [Deep Residual Learning for Image Recognition]()(ResNet)
  + https://arxiv.org/abs/1512.03385
* [x] [Sample-level CNN Architectures for Music Auto-tagging Using Raw Waveforms]() (SampleCNN)
  + https://arxiv.org/abs/1710.10451
* [x] [Custom 2D CNN]() (4Layer 2D-CNN)
* [x] [Custom 2D CNN + GRU]() (4Layer 2D-CNN + GRU)


# 问题
* UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().

RNN的权值并不是单一连续的，这些权值在每一次RNN被调用的时候都会被压缩，会很大程度上增加显存消耗。


