
# 音乐
音乐能激活大面积[神经](https://www.iiiff.com/article/3947)。

# 听觉
伏隔核（奖赏回路的核心，尾壳核内下方，前方为嗅前核，后面为纹状核）与受试者对一段音乐的喜爱程度直接相关。
每个人的生活体验不同，就会[喜欢不同音乐]((https://www.iiiff.com/article/178))。
基本的音乐元素是熟悉的，容易产生联想，对陌生音乐产生某种预期。


# 深度神经网络
## 建模
A1(41) -> A2(42) -> A3(22) -> T2/T3(21,20)

## VGGish-BiGRU网络
[VGGish-BiGRU](https://blog.csdn.net/Sohu564/article/details/123976967) 

[VGGish-BiGRU pytorch实现](https://github.com/azarmehri/lung-sound-vggish) 


# 实验
## todo
* 深度模型的训练和激活抽取
* fMRI的ROI抽取
* 训练解码模型
* 在测试集中进行解码并计算皮尔逊相关系数

增强8倍后的测试集（音频顺序记录在 `run_dataset/test_order_in_DNN.txt` 中）
```
Test_run-01-05_metal.00039_15s6.npy
```
其中，`01` 表示 `ExpValOrder.mat` 矩阵中的列号（即Run-id），`05` 表示矩阵中的行号，可以查找到对应的音频刺激。

激活保存在 `speedup/_classifier_0_feats.npy` 中：

(240x8)x128x128 -> 15x128x512 （15个batch, 每个batch 128 个音频，每个音频的特征长度为 512）

## 解码
在测试集中，使用线性解码将fMRI转到 特征向量（和深度网络的激活进行相关性计算）。

## 分类
[使用可解释的技术进行音频分类](https://ww2.mathworks.cn/help/deeplearning/ug/investigate-audio-classifications-using-interpretability-techniques.html)
 
## 运行

1. 准备音频数据
```maltab
music_genre_fMRI/process_music.m
```
2. 训练音频分类网络
```matlab
genre_classification.m
```

3. 提取对应第2步中输入音频数据时的大脑激活，将训练和测试特征保存为`sub-001.mat`
```matlab
get_fMRI_activation.m  % 做之前需要用SPM根据music_genre_fMRI/readme.md的步骤进行处理?
music_genre_fMRI/get_fMRI_intensity.m  % 这个不需要，只要get_fMRI_activation.m?
```


4. 后台运行 matlab 任务进行大脑激活和模型激活的回归和预测
```matlab
mr analysis_FeaturePrediction &
```

## 结果
最大类脑相似度：0.266918。


## 声源高度定位实验
[数据](https://openneuro.org/datasets/ds004256/versions/1.0.5) 


# 听觉通路
![听觉通路](https://wimg.iiiff.com/wimg/0908/cross-section-of-auditory-cortex.jpg)

## 耳蜗核（Cochlear nucleus）
[耳蜗核](https://zh.wikipedia.org/wiki/%E8%80%B3%E8%9C%97%E6%A0%B8)分为腹侧耳蜗核（Ventral Cochlear Nucleus，VCN）和背侧耳蜗核（Dorsal Cochlear Nucleus，DCN）。

### 腹侧耳蜗核（VCN）
VCN 约有 30000 个细胞，分为腹前耳蜗核（Anteroventral cochlear nucleus，AVCN）和腹后耳蜗核（Postventral cochlear nucleus，PVCN）。

#### AVCN 
主要细胞类型为多毛细胞（Bushy cell）和多极细胞（Multipolar cells）；

球形多毛细胞（Spherical Bushy Cell，SBC）：几何形状像球体，连接内侧上橄榄复合体；

球状多毛细胞（Globular Bushy Cell，GBC）：大致是球状的，连接外侧上橄榄复合体。

#### PVCN 
主要为多极细胞和章鱼细胞（Octopus cells）。

### 背侧耳蜗核（DCN）
DCN 约有 12000 个细胞。



## 上橄榄（Superior olive）

## 下丘（Inferior colliculus）

## 内侧膝状体核（Medial geniculate，MGN）



# 背侧通路
[使用深度学习训练 3-D 声音事件定位和检测](https://ww2.mathworks.cn/help/audio/ug/train-3d-sound-event-localization-and-detection-seld-using-deep-learning.html) 






# 听觉皮层

## 解剖信息
位于颞上回（并延伸到外侧沟和颞横回），大致位于41区、42区以及部分22区。
41区域和42区是A1的一部分。
41区也称为前颞横回(H)，其内侧由岛旁区域52(H)界定，外侧由后颞横回42(H)界定。
42区称后颞横回，内侧由前颞横回界定，外侧由上颞区22界定。

螺旋选择性神经元位于原发性听觉皮质前外侧边缘附近的皮层区域。
在人类的最近的功能成像研究中也已经确定了这个位置的[选音区域](https://zh.wikipedia.org/wiki/%E5%90%AC%E8%A7%89%E7%9A%AE%E5%B1%82)。

### 颞横回 HG 
[颞横回](https://zh.wikipedia.org/wiki/%E9%A2%9E%E6%A8%AA%E5%9B%9E) （英语：Transverse temporal gyrus）是大脑外侧沟内初级听觉皮层的脑回，是颞叶的一部分，位于布罗德曼系统第41、42分区。
颞横回又被称为黑索氏（赫氏）回（Heschl's gyrus），以纪念奥地利解剖学家理查德·L·黑索。

颞横回是颞叶的一部分，位于颞平面（负责语言生成）之上，但与之相隔。颞横回实际上是一组脑回，在左右大脑半球中数量不一。
颞横回是内外走向（朝向脑部中心，所以叫做横，横亘在脑中），而其它脑回是前后走向。

颞横回是最先处理听觉信息的大脑皮层结构，属于初级听觉皮层，是音高（频率高，则音高）知觉的核心脑区。
该区域是来自内侧膝状体（丘脑的特定听觉中继核）的听觉信息的主要皮质投射区域。


### 颞横沟 HS


### 颞平面 
[颞极平面](http://conxz.net/2017/10/22/planum-temporale-asymmetry/) （planum temporale, PT）是位于颞上回（superior temporal gyrus）后部背侧皮层的一个三角形区域（听觉皮层的正后方，外侧沟的结构内），其左侧脑区与经典的语言功能区 Wernicke’s area存在一定的重合。

## 处理
输入：初级听觉皮层从接收的直接输入内侧膝状核的的丘脑和因此被认为是识别音乐的基本元件，例如音高和响度。

音调：[罗氏假性前额叶皮质](https://zh.wikipedia.org/wiki/%E5%90%AC%E8%A7%89%E7%9A%AE%E5%B1%82)（RMPFC）



## 特点

皮层中的相邻细胞响应相邻的频率。

单个细胞始终得到激发的声音在特定频率，或频率的倍数。

听觉皮层的个体差异很大，成年人的初级听觉皮层（A1）所拥有的功能特性高度取决于幼年时期接触到的声音。

 单侧听觉皮层受损会导致轻微的听力损失，而双侧都受到破坏可能会导致皮质性耳聋。

听觉皮层对声音和语音的信息处理是[并行](https://zhuanlan.zhihu.com/p/403326770)的。

[反馈](https://zh.wikipedia.org/wiki/%E5%90%AC%E8%A7%89%E7%9A%AE%E5%B1%82)：除了通过听觉系统的较低级部分接收从耳朵输入的信号外，听觉皮质还可以将信号发送回这些区域，并与大脑皮质的其他部分相互连接。

听觉皮层对伽马波段的声音有不同的反应。当受试者暴露于40 赫兹的三或四个周期时，EEG数据中出现异常尖峰，这不是其他刺激物质。

伽玛带激活（25至100 Hz）已被证明是在感知事件感知和认知过程中存在的。


# 初级听觉皮层（A1）
核心区，位于上颞回。
从 MGB 的腹侧部接受点对点的输入。

在声音刺激开始后的10毫秒至几百毫秒的时间内（初始期），A1中会有大量神经元放电，表现为密集编码（大脑知道有声音来了，选择性较低）；

然后，随着声音刺激的持续（持续期），A1中的放电神经元数量减少，表现为稀疏编码（对声音进行识别，这时选择性较高）。


# 次级听觉皮层（A2）
带状区（belt or peripheral area）紧靠着A1，头/腹侧-->颞叶的头/腹侧；尾测-->颞叶的背/尾侧。

从 [MGB 的带状区](https://www.ncbi.nlm.nih.gov/books/NBK10900/) 接受更多的扩散输入，因此在它们的音调组织中回不大精确。


# 第三级听觉皮层（A3）
伞状区（parabelt）和带状区的侧面相邻。

# 中/下颞（T2/T3）
superior (T1), 
middle (T2), 
inferior (T3) gyri,
fusiform gyrus (T4),
parahippocampal gyrus (T5)

# 论文
[分析 fMRI 的论文](/data2/whd/win10/doc/paper/doctor/doctor.Data/PDF/1580343150)

“听觉通路突触连接的经验和细化”位于神经科学的第49章图49-17。


# 数据
[10个音乐流派](https://openneuro.org/datasets/ds003720/versions/1.0.0)
注意：sub-001的原始.nii文件有4个前缀重复了，需要重命名。

[听富有情感的音乐EEG](https://openneuro.org/datasets/ds002721/versions/1.0.2)

[听富有情感的音乐EEG-fMRI](https://openneuro.org/datasets/ds002725/versions/1.0.0)
