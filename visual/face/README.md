# 思路
1. 找出随机初始化参数模型自带人脸识别的原因是否是模型结构的原因（和全连接层的分类网络进行对比）;random feedforward connections (genes?)；
使用其他网络进行测试，看是否只对人脸有特殊反应；
是否只有特定的网络结构才会产生对人脸的特殊反应；
不同的类别(模式)对不同的网络架构会产生特殊的反应(结构对功能有偏好)?


2. 根据CorNet设计类脑人类识别网络；
测试随机初始化的CorNet是否也对人脸有特殊的反应；

3. 比较类脑网络和人脑的激活相似性

4. inverted face images are significantly lower than those to upright faces 

5. 天生看脸,喜欢看均匀脸(省能量?舒服?合理?颜值即舒服);
那不同文化喜欢不同的脸又是什么原因?

6. 人在做人脸（或比如文字等比较难的目标）识别时回自动做矫正。

7. 微型终板电位在没有动作电位的情况下自发进行量子递质释放。
单个 ACh 受体通道对基本电流的电压响应仅为大约 0.3 μV，
随机开放5000中的2000个ACh通道，形成0.5mV微型终板电位。
动作电位（75mV）释放150个量子递质，每个0.5mv

8. 在颞极中区分熟悉的人脸和陌生的人脸


# 数据
[面部处理](https://openneuro.org/datasets/ds000117/versions/1.0.5) ：包含刺激。

[面部编码](https://openneuro.org/datasets/ds000232/versions/00001)


# 技巧
## 画图
[export_fig](https://www.modb.pro/db/102829)

# 参考论文
## 未训练的网络可以进行面部检测 
"Face Detection in Untrained Deep Neural Networks" </br>

论文位于 Ubuntu 的本地目录
```
/data2/whd/win10/doc/paper/doctor/doctor.Data/PDF/3865877311
```

补充材料位于：
```
/data2/whd/win10/doc/paper/doctor/doctor.Data/PDF/3997157088
```

Seungdae Baek, Min Song, Jaeson Jang, Gwangsu Kim, and Se-Bum Paik*

*Contact: sbpaik@kaist.ac.kr


### 1. System requirements
- MATLAB 2019b or later version
- Installation of the Deep Learning Toolbox (https://www.mathworks.com/products/deep-learning.html)
- Installation of the pretrained AlexNet (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)
- No non-standard hardware is required.
- Uploaded codes were tested using MATLAB 2019b.

### 2. Installation
- Download all files and folders. ("Clone or download" -> "Download ZIP")
- Download 'Data.zip', 'Stimulus.zip' from below link and unzip files in the same directory
- [Data URL] : https://doi.org/10.5281/zenodo.5637812
- Expected Installation time is about 45 minutes, but may vary by system conditions.
 
### 3. Instructions for demo
- Run "Main.m" and select result numbers (from 1 to 6) that you want to perform a demo simulation.
- Expected running time is about 5 minutes for each figure, but may vary by system conditions.

### 4. Expected output for demo
- Below results for untrained AlexNet will be shown.
  - Result 1) Run_Unit: Spontaneous emergence of face-selectivity in untrained networks (Fig.1, Fig.S1-3)
  - Result 2) Run_PFI: Preferred feature images of face-selective units in untrained networks (Fig.2, Fig.S4) 
  - Result 3) Run_SVM: Detection of face images using the response of face units in untrained networks (Fig.3, Fig.S11-12)  
  - Result 4) Run_Trained: Effect of training on face-selectivity in untrained networks (Fig.4) 
  - Result 5) Run_Invariance: Invariant characteristics of face-selective units in untrained networks (Fig.S5) 
  - Result 6) Run_View: Viewpoint invariance of face-selective units in untrained networks (Fig.S8)

### 参考
[原始库](https://github.com/KamitaniLab/GenericObjectDecoding)





## 面孔知觉 
《认知神经科学 关于心智的生物学》
第六章 物体识别 -> 面孔知觉

### 面孔知觉的神经机制
拥有特异性面孔加工模块。
梭状回面孔识别区（fusiform face area, FFA）。
面孔刺激呈现后170ms引发一个巨大的负 EEG 信号，即N170（N170 response）。

关于面孔的倾向反映了根植于我们进化史的行为。


### 面孔物体知觉的分离
面孔倒立效应


# 工具
[面部行为分析工具](https://github.com/TadasBaltrusaitis/OpenFace) 


