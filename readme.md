
# 研究内容
## 愿景
```

如果你愿意一层一层

一层的剥开我的心

你会鼻酸

你会流泪

-- 五月天, 2008

```

欢迎来到寻找自己的旅途！ 

## 路线
认知  --> 动作 --> 形态 

AI --> NeuroAI --> Organoid Intelligence（类器官智能）
图灵测试 --> 俱身图灵测试

## 输入

### 视觉
包括人脸、跟踪

### 融合
听觉、多模态融合

## 意识
## 记忆
记忆内容的精细化解码。

## 定位
《神经科学原理》5.1.3 可以解码海马空间认知图来推断位置

## 预测

## 强化学习
强化学习的神经基础。

Reinforcement Learning, An Introduction 神经科学章节。

神经科学原理（第六版）第43章 动机、奖励和上瘾状态。


## 输出
### 初级运动皮层




# 大脑模型
寻找意识的神经相关物甚至意识的物质基础。

从大脑（或者生命）进化的角度推导大脑的制造过程；
或者从成品的角度对现有大脑做逆向工程。

## 刺激、激活、意识
输入、中间层、输出（视觉、听觉、情绪）。

### 统一
[视频的情绪表示](https://github.com/KamitaniLab/EmotionVideoNeuralRepresentation)。

代码位于 `visual/emotion/` 目录下。

### 激活 -> 刺激
[根据 fMRI 重建图像](https://github.com/KamitaniLab/DeepImageReconstruction)

代码位于 `visual/reconstruction` 目录下。

### 刺激 -> 激活 
预测大脑激活。

使用生成网络 VAE，将图片作为输入，预测 `fMRI` 的激活响应。（可以基于visual/object_decoding 做）

### fMRI -> 意识
[根据 fMRI 解码出目标类别](https://github.com/KamitaniLab/GenericObjectDecoding)

[解码空间位置信息](https://github.com/KamitaniLab/PositionDecoding) 


### 意识 -> fMRI
参考图像生成网络。

## 多模态融合的类脑网络
设计一个多模态融合网络以在线预测大脑的激活，以克服多个模态之间相互影响的问题（这很可能是导致单一模态预测精度低的原因）。

人有五感（分别为视、听、味、嗅、触），但由于是在电影《阿甘正传》数据集上进行实验，主要考虑图像（视频）、声音（音轨）和文本的输入（其他的输入为静息状态）。

[DeepMind 多模态多任务学习](https://baijiahao.baidu.com/s?id=1732695949873185896&wfr=spider&for=pc) 


## 存在的挑战
* 生成网络：生成一个三维点云分布（生成对抗网络）
* 声音信号和图像信号的融合：不是输入时候的融合
* 听觉通路的建模
* 除了视觉、听觉、语言之外其他脑区的建模

## 计划
* 随机初始化人脸识别模型能够对人脸有相对于其他物体有特殊反应的原因是否为网络模型架构

1. 如果用全连接模块应该就是随机的了；
2. 不同模型架构效果可能不一样；


* 人脸识别时的大脑激活和深度模型激活相似

1. 找大脑对人脸的激活数据；
2. 人脸识别模型：能否用Cornet或者其他；



* 先实现背侧通路和腹侧通路的融合

1. 将卷积用Coenet中的模块替换（已经有Cornet的tensorflow的实现）




* 听觉通路神经网络实现
1. 搞清楚几个听觉皮层之间的关系，处理音乐是否有次级听觉皮层更好的皮层；
2. 输入到初级听觉皮层的是什么信号（经过了什么处理，是否可以用一层替代）；
3. 对应深度网络的几个模块；


# 进展

* [音乐风格分类](https://github.com/mmoksas68/Music-Genre-Classification)

* [学习音频+视频+语言表征](https://avlnet.csail.mit.edu/)

* [音频+视频 场景感知对话](https://github.com/ictnlp/DSTC8-AVSD)

* [视频语言表示](https://github.com/linjieli222/HERO)

* [3D点云生成PointFlow](https://github.com/stevenygd/PointFlow)

* [生成各种点云的工具](https://github.com/OpenDroneMap/ODM)

* [自编码点云生成](https://github.com/optas/latent_3d_points)
python 2.7, tensorflow 1.0


## 理解

* 视频：背侧流+腹侧流

* 音频：初级听觉皮层（A1）、次级听觉皮层

* 语言：韦尼克区（颞上回、颞中回后部、缘上回、角回）：语言理解

## fMRI
一次全脑扫描时间TA：一次全脑扫描第一层与最后一层时间间隔TA=TR-TR/(Number of Slices)

# 目录结构

* config.m      项目的配置文件
* init_env.m    初始化必要的变量和环境
* memory/spatial_memory_pipeline   进行导航时空间感知的本体表征转换为全局表征


# 研究计划内容

## 计划内容
该课题主要包括三个阶段：类脑图像识别的设计分析和建模、类脑跟踪的机理研究和仿真、类脑感知融合探索和研究。
第一阶段，根据大脑皮层视觉中腹侧流通路的研究，如图1所示，涉及目标识别的通路主要包括初级视觉皮层V1、纹外皮层V2和V4、颞枕皮层TEO、下颞皮层IT，其中下颞皮层时目标识别的主要区域。参照腹侧流通路的解剖结构，利用卷积和循环结构，构建大脑对齐的深度模型，可以设计相对应的类脑图像识别网络，并获得输入图像在深度网络中的激活响应，研究图像识别功能在大脑皮层和深度神经网络中表现的激活相似性，同时进行进行神经度量和行为度量，利用类脑识别分数进行深度神经网络和大脑的对比，并利用设计并训练好的类脑模型，确定识别的图像模式分别在深度网络和大脑皮层中的位置，寻找图像识别的类别信息在深度网络中的表征模式与大脑中的激活特征之间的映射关系。

考虑到人类关注面孔的倾向根植于人类的进化史，而不是新进获得的文化传统，并且已证实面孔知觉不是采用跟物体识别相同的加工机制，而是采用特异性面孔加工模块。
梭状回面孔区在面孔知觉中起到了重要的作用，相比于识别其他类型的物体，对人的面孔反应强度更大。

第二阶段，参照人类大脑皮层中背侧流通路进行类脑跟踪的机理研究和仿真，通过量化的大脑相似性分数，并利用所获得的皮层解剖知识来启发类脑跟踪模型的设计。如图2所示，类脑跟踪模型包括映射到人脑的四个区域：初级视觉皮层，中颞和上颞内侧区、额叶视区和脑干/小脑 。初级视觉皮层使用经典的卷积层进行建模，执行预处理以减少数据大小。中颞区和上颞内侧区使用动态滤波器网络进行建模，额叶视区使用循环神经网络进行建模。对于最开始的输入刺激，神经网络的输出表示大脑对齐模型中的深度神经网络的激活响应和边界框，而大脑皮层在跟踪时的表示是大脑皮层的激活和眼睛注视的位置，同时类脑跟踪模型显示了计算机视觉跟踪性能与大脑跟踪响应之间的相似性关系。为了比较模型，通过检查深度神经网络中各层的激活来构建到皮层的映射，以便能够很好地理解特定大脑皮层区域中的激活，理想情况下，这种大脑皮层激活不需要多余参数的类脑模型所预测得到，将会降低传统深度跟踪模型的冗余度。因此类脑跟踪模型由卷积层、动态滤波网络、长短时记忆网络和全连接层四个神经网络模块组成，它们类比于大脑平滑跟踪皮层通路中的初级视觉皮层、中颞区和上颞内侧区、额叶视区、脑干/小脑，其中脑干/小脑是运动预测器，将额叶视区的输出转换为相应的运动响应。这种明确的大脑分区思想是设计类脑跟踪模型重要的一步，并且致力于寻找更通用的网络结构。整个模型包括在皮层区域没有差异的神经网络，以及各种改进类脑跟踪模型的连接。并在此基础上对比激活和行为之间的相似性。

同时时在类脑跟踪模型的基础之上分析模型的动态响应和大脑皮层对运动的响应之间的相似性或相关性。并加入第一阶段类脑图像识别模型，类比于大脑皮层中腹侧流和背侧流之间的相互连接和影响，进一步提高视觉皮层的建模精度，以便更好地适应真实和复杂的动态场景，提高类脑模型的跟踪精度和类脑的真实性。

第三阶段，在之前工作的基础之上，考虑大脑中其他模态的信息和视觉信息的相互作用，类比于人类感知中的“通感”。考虑在人所接收的所有感知信息中，听觉信息是仅次于视觉信息的第二大感知源，两者占人所有感知信息的百分之九十以上，所以在建模类脑的多模态感知时主要考虑人类的听觉。听觉识别通路主要包括初级听觉皮层A1、带状区Belt、伞状区PB、中颞/下颞区，并利用卷积神经网络和循环网络为基本结构，模拟人脑处理模式，构建大脑对齐的类脑模型，进行激活和行为之间的对比，包括大脑激活和神经网络激活之间的相似性、网络预测的音频类别和人类动作选择之间的相似性。

同时由于听觉识别的核心区域中颞/下颞区和第一阶段中的目标识别的下颞核心区有重叠部位，为视觉和听觉的融合处理提供了神经解剖基础，可以进一步在图像识别模型和音频识别模型的基础之上设计视觉/听觉融合模块，进行高层特征级别融合，有望进一步提高处理复杂环境输入的能力和提升类脑模型的预测能力。
在进行类脑模型的研究过程中，期待逐步建立一套科学合理的类脑评价指标用于评价所设计的类脑深度神经网络模型和人脑大脑处理信息的相似程度，在类脑图像识别中表现为图像分类和人进行图片识别时选择的相似性，在类脑跟踪中表现为深度神经网络输出的定位框和以人眼中央凹为中心的注意力范围，在感知融合中表现为在融合其他类型信息的复杂条件下模型的预测输出和人行为的相似性。该指标不仅衡量和模型的预测输出和人类行为之间的相似性，而且还衡量模型面对相同的输入刺激，各个中间层的激活响应和人类大脑皮层各个区域的激活（血液动力学响应或者脑电信号等）之间的相似性或相关性，实现从模型结构、皮层激活响应、人类行为等方面进行综合的类脑相似性解释。

## 研究基础
关于该课题三个阶段已有一定的研究基础，包括基本的类脑图像识别模型和评价指标，可以要在此基础上基于已有的人脸识别刺激数据和对应的多模型神经成像数据，利用人脸识别任务分析类脑模型和深度神经网络对相同图像刺激所产生的激活之间的相似性或关联关系，以及进一步分析负责人脸识别区的梭状回和负责目标识别区的下颞回之间的联系和区别。

本课题组已设计和测试类脑跟踪中背侧通路的建模和仿真，建立初步的运动感知模型，定量化验证中颞区和上颞内侧区和动态滤波网络激活的相似性，并在公开数据集上进行了仿真和验证，可以在此基础上加入图像识别的腹侧流模型，进一步研究背侧流和腹侧流共同作用下动态环境的建模和仿真。

类脑感知融合研究中已对类脑听觉皮层模型进行了初步建模，在音乐流派分类数据集合对应的核磁共振数据集上进行了响应分析，需要在此基础上分析深度模型激活和大脑皮层激活之间的相似性，并加入第一阶段腹侧流通路的建模，以进一步提高类脑模型的鲁棒性，提升模型的激活和行为的预测精度。


## 研究意义
该类脑研究是可解释深度神经网络的一个很好地解决方案，设计大脑皮层对齐的类脑模型不仅可以设计出符合生物学解剖规律的深度神经网络，利用神经科学的原理来解释所设计的深度网络模型，有望解决深度神经网络“黑盒”的问题，还可以精简模型的层数，减少模型冗余，有利于工程化部署，加快计算机视觉算法的产业化落地。     

同时利用设计并训练好的类脑深度神经网络探索脑机接口的应用和人产生视觉概念乃至意识的神经相关物，利用已经比较成熟的高性能计算机和深度学习算法，解决神经科学实验条件受限的不足，解决深度学习领域模型的复杂度越来越高、可解释性越来越差的问题。


## 相关期刊
[Nature](http://www.letpub.com.cn/index.php?journalid=6054&page=journalapp&view=detail) 

[Nature Machine Intelligence](http://www.letpub.com.cn/index.php?journalid=11172&page=journalapp&view=detail) 
一区 与神经科学交叉  每月收2-4篇

[Nature Communications](http://www.letpub.com.cn/index.php?journalid=8411&page=journalapp&view=detail)
每篇版面费为5200美金,$5,790 (The Americas*, Greater China** and Japan)

[Nature Computational Science]
未收录

[AI](http://www.letpub.com.cn/index.php?journalid=892&page=journalapp&view=detail) 
二区

[TIP](http://www.letpub.com.cn/index.php?journalid=3390&page=journalapp&view=detail) 
一区

[Cognitive Computation](http://www.letpub.com.cn/index.php?journalid=8477&page=journalapp&view=detail)
二区

[Computational Intelligence and Neuroscience](http://www.letpub.com.cn/index.php?journalid=9736&page=journalapp&view=detail)
三区

[IJCV](http://www.letpub.com.cn/index.php?journalid=3703&page=journalapp&view=detail)  
二区


# 安装

## 进化树
[phylip](https://evolution.genetics.washington.edu/phylip/) 

    #显示目前conda的数据源有哪些
    conda config --show channels
    ​
    #添加数据源：例如, 添加清华anaconda镜像：
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --set show_channel_urls yes

    conda install pytorch=1.0.1 torchvision cudatoolkit=10.1 -c pytorch -y



Matlab
Solve "VideoReader" function problem (Could not read file due to an unexpected error. Reason: Unable to initialize the video obtain properties)

https://www.cnblogs.com/walker-lin/p/11520064.html


安装 [Tensorflow v1.14.0] and the following dependencies

    numpy 1.11.3 -> 1.14.6
    /home/d/anaconda2/envs/hart/bin/pip 
    ./pip install tensorflow (1.14.0)
    ./pip install tensorflow==1.2.0 (need libcusolver.so.8.0)
    No module named core_rnn_cell_impl (The code is updated to tensorflow 1.3.0.)
    
    Could not dlopen library 'libcudart.so.10.0';  Cannot dlopen some GPU libraries. Skipping registering GPU devices..
    export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
    source ~/.bashrc 
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    sudo chmod a+r /usr/local/cuda/include/cudnn.h
    sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    cd /usr/local/cuda/lib64
    sudo ln -sf libcudnn.so.7.6.3 libcudnn.so.7 (libcudnn.so.7 is not a symbolic link)
    sudo ln -sf libcudnn.so.7 libcudnn.so
    sudo ldconfig
    sudo ldconfig /usr/local/cuda-10.0/lib64/  (symbolic, force)

 (using `pip install -r requirements.txt` (preferred) or `pip install [package]`):
* matplotlib==1.5.3
* numpy==1.12.1
* pandas==0.18.1
* scipy==0.18.1

    
    predictive pursuit
    Python:
    /data2/whd/workspace/sot/CORnet/venv/bin/python
    /data2/whd/workspace/sot/CORnet/venv/bin/pip
    
    实验室主页：http://dicarlolab.mit.edu/
    
    brain-score
    先安装： result_caching, brainio_collection, brainio_base
    /data2/whd/workspace/sot/CORnet/venv/bin/python ./setup.py install
    
    /data2/whd/workspace/sot/CORnet/venv/bin/pip install xarray==0.12
    botocore

* 安装Maltab相关工具

Q： Undefined function or variable 'MRIread'.

A： 安装工具包freesurfer：https://github.com/fieldtrip/fieldtrip/tree/master/external/freesurfer
    ```
    https://github.com/fieldtrip/fieldtrip/archive/refs/tags/20211209.tar.gz
    ```


## Data

1. [SPM demo data](ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson/wakemandg_hensonrn/)

匿名登录：ftp ftp.mrc-cbu.cam.ac.uk

用户名：anonymous
密码：为空（直接回车）

wget ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson/wakemandg_hensonrn/* --ftp-user=anonymous --ftp-password= -r
    
1. Download KITTI dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). We need [left color images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) and [tracking labels](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip).
2. Unpack data into a data folder; images should be in an image folder and labels should be in a label folder.
3. Resize all the images to `(heigh=187, width=621)` e.g. by using the `scripts/resize_imgs.sh` script.
4. Download [BrainIO](https://github.com/brain-score/brainio_collection/blob/master/brainio_collection/lookup.csv) data

## Training

1. Download the AlexNet weights:
    * Execute `scripts/download_alexnet.sh` or
    * Download the weights from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put the file in the `checkpoints` folder.
2. Run

        python scripts/train_hart_kitti.py --img_dir=path/to/image/folder --label_dir=/path/to/label/folder

The training script will save model checkpoints in the `checkpoints` folder and report train and test scores every couple of epochs. You can run tensorboard in the `checkpoints` folder to visualise training progress. Training should converge in about 400k iterations, which should take about 3 days. It might take a couple of hours between logging messages, so don't worry.

## Evaluation on KITTI dataset
The `scripts/eval_kitti.ipynb` notebook contains the code necessary to prepare (IoU, timesteps) curves for train and validation set of KITTI. Before running the evaluation:
* Download AlexNet weights (described in the Training section).
* Update image and label folder paths in the notebook.

##  Freesurfer
1. Install

        chsh -s /bin/tcsh
        
        d:/data2/whd/software/freesurfer> setenv FREESURFER_HOME /data2/whd/software/freesurfer/
        d:/data2/whd/software/freesurfer> source $FREESURFER_HOME/SetUpFreeSurfer.csh
        
        -------- freesurfer-linux-centos6_x86_64-7.1.1-20200723-8b40551 --------
        Setting up environment for FreeSurfer/FS-FAST (and FSL)
        FREESURFER_HOME   /data2/whd/software/freesurfer/
        FSFAST_HOME       /data2/whd/software/freesurfer//fsfast
        FSF_OUTPUT_FORMAT nii.gz
        SUBJECTS_DIR      /data2/whd/software/freesurfer//subjects
        INFO: /home/d/matlab/startup.m does not exist ... creating
        MNI_DIR           /data2/whd/software/freesurfer//mni

2. Usage

        setenv SUBJECTS_DIR /data2/whd/workspace/sot/hart/result/freesurfer/
        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
        export $LD_LIBRARY_PATH
        
        /data2/whd/software/Qt5.12.9/5.12.9/gcc_64/lib
        
        Cannot load library /data2/whd/software/freesurfer/lib/qt/plugins/platforms/libqxcb.so: (/data2/whd/software/freesurfer/bin/../lib/qt/lib/libQt5Core.so.5: version `Qt_5.12' not found (required by /usr/lib/x86_64-linux-gnu/libQt5XcbQpa.so.5))
QLibraryPrivate::loadPlugin failed on "/data2/whd/software/freesurfer/lib/qt/plugins/platforms/libqxcb.so" : "Cannot load library /data2/whd/software/freesurfer/lib/qt/plugins/platforms/libqxcb.so: (/data2/whd/software/freesurfer/bin/../lib/qt/lib/libQt5Core.so.5: version `Qt_5.12' not found (required by /usr/lib/x86_64-linux-gnu/libQt5XcbQpa.so.5))"

## Polyspace

    polyspace-access -host d -port 9443 -create-project testProject


## Citation

If you find this repo useful in your research, please consider citing:

    @inproceedings{Kosiorek2017hierarchical,
       title = {Hierarchical Attentive Recurrent Tracking},
       author = {Kosiorek, Adam R and Bewley, Alex and Posner, Ingmar},
       booktitle = {Neural Information Processing Systems},
       url = {http://www.robots.ox.ac.uk/~mobile/Papers/2017NIPS_AdamKosiorek.pdf},
       pdf = {http://www.robots.ox.ac.uk/~mobile/Papers/2017NIPS_AdamKosiorek.pdf},
       year = {2017},
       month = {December}
    }


# 学习
## 视频
[大脑与认知科学](https://www.bilibili.com/video/BV1Xb4y1z7eR/?p=1&vd_source=98260b8dbf6f69741edcee62e52758ab) 



# 参考
## 书籍
1. 探索脑（全文背诵）、神经科学原理（只抽出研究的章节看）
2. 认知神经科学
3. 人的意识
4. Reinforcement Learning, An Introduction
