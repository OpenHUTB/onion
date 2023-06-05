# 通用目标解码
利用 fMRI 激活来预测类别模式。 该仓库包括 [论文](https://www.nature.com/articles/ncomms15037) 的数据和演示代码。
通用目标解码方法能解码任意目标类别（包括再模型训练过程中没有使用的类别）。


相关数据和依赖软件的[百度网盘链接](https://pan.baidu.com/s/1aKY7aP0ggQasj9ky9xiwZA) ，提取码：dong。
```text
images.zip                 为实验所用到的图像
ImageFeatures.mat          为图像输入深度模型时的激活
Subject1-5.mat             为受试看图像时的大脑激活

preprocessed_fMRI_features 为五个受试已处理的fMRI数据和视觉特征（CNN1-8, HMAX1-3, GIST, and SIFT），
ds001246-download.tar.gz   为未处理的fMRI数据

BrainDecoderToolbox2-0.9.17 解码工具箱
spr_1_0

```
fMRI数据使用[BrainDecoderToolbox2](https://github.com/KamitaniLab/BrainDecoderToolbox2) 和 [bdpy](https://github.com/KamitaniLab/bdpy) 进行处理。

该研究中用于高层定位实验的刺激图像可以通过 [链接](https://forms.gle/c6HGatLrt7JtTGQk7) 进行获取。

一些测试图片是从ILSVRC 2012的训练图片中获取，位于[data/stimulus_info_ImageNetTest.csv](https://github.com/KamitaniLab/GenericObjectDecoding/blob/master/data/stimulus_info_ImageNetTest.csv) 。


# 通用解码示例

这是 `Matlab` 的通用解码示例代码。

## 数据组织

所有的数据都放在 `workDir`(init.m) 的目录中。
数据目录中应该有下列文件：

    workDir/--+-- Subject1.mat (fMRI数据, 受试 1)
            |
            +-- Subject2.mat (fMRI数据, 受试 2)
            |
            +-- Subject3.mat (fMRI数据, 受试 3)
            |
            +-- Subject4.mat (fMRI数据, 受试 4)
            |
            +-- Subject5.mat (fMRI数据, 受试 5)
            |
            +-- ImageFeatures.mat (使用Matconvnet提取的图像特征)


## 分析

在Matlab中依次运行下列脚本。

```matlab
>> analysis_FeaturePrediction          % 特征预测（10个CPU运行2天以上）
>> analysis_FeaturePredictionAccuracy  % 特征预测精度
>> analysis_CategoryIdentification     % 类别识别
>> createfigure                        % 绘制 特征解码/类别识别 精度图
>> convert_decodedfeatures             % 为深度图像重建 转换 特征预测结果
```

所有的结果会保存在 `results` 目录中，当前处理的文件保存在文件锁`tmp`目录中。

为了可视化结果，运行下列脚本。

```
>> createfigure
```

`createfigure.m` 会创建两个图像：一个显示图像特征和平均类别特征预测的结果，
另一个显示类别识别类别识别的结果。
这些图像会以PDF格式保存在 `result` 目录中（`FeaturePredictionAccuracy.pdf` 和 `IdentificationAccuracy.pdf`）。


## 工具箱说明

- [BrainDecoderToolbox2](https://github.com/KamitaniLab/BrainDecoderToolbox2)
下载后放置在 `matlab\software\matlab_utils\BrainDecoderToolbox2-0.9.17` 目录中

- [SPR](https://bicr.atr.jp//cbi/sparse_estimation/sato/VBSR.html)
安装 1.0 版本的 SPR（放置在`matlab\software\matlab_utils\SPR_2011_1111`），代码和.c文件放在同一目录下，并用mex_compile进行重新编译。
注意：高版本matlab需要在 `mex_compile.m` 中添加兼容性数组维度，否则出现“请求的数组超过预设的最大数组大小”错误。
```matlab
mex -v weight_out_delay_time.c '-compatibleArrayDims'
```

- [Matconvnet](https://www.vlfeat.org/matconvnet/install/) 
编译步骤：进入`matconvnet`所在的目录，运行
```commandline
addpath('matlab')
vl_compilenn
```
执行`vl_compilenn`编译错误`Unable to find cl.exe`
将目录`C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.21.27702\bin\Hostx64\x64`添加到环境变量中。

# 其他
降低深度模型层和对应脑区的激活之间的损失。

# 问题
如果运行完`analysis_FeaturePrediction.m`后，结果文件中每个受试的总文件大小不是`2.17 GB (2,340,685,118 字节)`，而是`224 MB (235,368,488 字节)`的话，则原因是以下代码设置成了快速测试。
```markdown
numUnits = size(trainFeat, 2);
```

# 参考
[代码](https://github.com/KamitaniLab/GenericObjectDecoding) 