
# 音乐刺激
## 刺激下载

包含10中流派的音乐：爵士乐（jazz）、古典乐（classical）、嘻哈音乐（hip-hop）、迪斯科（disco）、乡村音乐（country）、忧郁布鲁斯歌曲（blues）、重金属音乐（metal）、流行乐（pop）、雷鬼乐（reggae）、摇滚乐（rock）。

[原始的音乐流派波形文件下载](http://marsyasweb.appspot.com/download/data_sets)

## 预处理音乐刺激数据
* 整个流程：process_music_fMRI.m
1. 对原始音乐视频进行预处理：CutInfo15s.m
2. 通过随机组合预处理的刺激构建实验刺激：MakeExpFile.m
3. 进一步正则化刺激：RMSnormalizer.m

### 数据集划分
训练集顺序保存在 `ExpTrnOrder.mat` 文件中，包括 40*12 个音频文件（.mat文件中所指的原始文件位于 `genres_wav15s` 文件夹中），1 列对应一个 fMRI 中的一个 run，实验的刺激文件位于 `genres_forExp` 文件夹中。

验证集顺序保存在 `ExpValOrder.mat` 文件中。

## 行为数据
在文件夹BehavData中，五个受试(sub-001,...,sub-005)都有对应的音乐风格类别判断的行为数据，第1列表示目标文件名，从第2列到第11列表示受试的回答（对应于10个音乐风格）。

presentation .sce文件 (MusicGenre_Run.sce)用于刺激的展示。


# fMRI说明

当5个受试（sub-001, ..., sub-005）听 10 个不同风格的音乐刺激时，使用 fMRI 来衡量大脑的激活。

整个fMRI数据目录包含受试自适应的子目录（sub-001, ...）。每个受试的目录包含下列子目录：
* anat：T1-weighted结构化成像。
* func：功能性信号（multi-band echo-planar images，多波段背景平面回波成像），EPI技术成像时间很短,可明显减少脉搏、不自主运动、肠蠕动、呼吸和心脏跳动等运动伪影。

## 数据格式描述
每个受试有18次运行，包含12个训练和6个测试。

### 任务描述数据
每个训练和测试数据都按照下面的标记进行分配：

训练数据：sub-00*_task-Training_run-**_bold.json

测试数据：sub-00*_task-Test_run-**_bold.json

### 事件记录数据
每个*_event.tsv文件包含一下信息：
* onset：刺激的开始
* duration：刺激持续时间
* genre：流派类别（10个流派）
* track：标识原始曲目的索引
* start：摘自原始曲目的开始时间（秒）
* end：摘自原始曲目的结束时间（秒）

所有刺激的持续时间都是15秒，使用2秒钟的淡入和淡出效果，整个信号强度都使用根均分误差进行正则化。

每次训练运行中，第一个刺激（1-15秒）和前一个运行（600-615秒）的最后一个刺激相同。

每次测试运行中，第一个刺激（1-15秒）和当前相同运行（600-615秒）的最后一个刺激相同。

*_event.tsv去除第1行（标题行）和第二行（重复刺激的行）后面的40行，对应于“ExpTrnOrder.mat”中的某一列（受试ID）的40行（40个音乐片段刺激）。

# 处理步骤

## 空间预处理

### 1. 重对齐（运动校正、头动校正）
对每个受试者所有的卷都和第一个EPI镜像对齐。
选择"Realign(Est & Res)"工具。
注：选择数据时候选择每个镜像卷的数目。
1. 选中"Data"，选择"New Session"，并选择创建"Session"选项；
2. 按下"Select Files"并，选中所有功能像（应该包含410个镜像（1:410)）；
3. 在"Reslice Options"中按下"Resliced images"，并选择"All Images + Mean Image"；
4. 保存成"jobs/religh.mat"并运行

重写对齐后的文件使用"r"开头（realign）：`rsub-001_task-Training_run-01_bold.nii`。
并生成平均镜像 `meansub-001_task-Training_run-01_bold.nii`。



### 2. 配准 
参考图像选择第一步对其的均值图像，源图像选择结构像图像，配准操作会修改结构像的头。选择配准工具"Coregister (Estimate)"
1. 选中"Reference Image"并选择再对齐步骤中生成的平均镜像"meansub-001_task-Training_run-01_bold.nii"；
2. 选中"Source Image"并选择结构像"anat/sub-001_T1w.nii"；
3. 保存成"jobs/coregister.mat"并运行。
4. 使用"Check Reg"工具查看配准结果，选择"Reference"为"func/meansub-001_task-Test_run-01_bold.nii"，"Source"为修改头后的结构像"anat/sub-001_T1w.nii"。

这一步会修改结构像"anat/sub-001_T1w.nii"的头。


### 3. 分割
1. 按"Segment"按钮；
2. 选中"Volums"字段，并选择前一步受试已经配准过的结构像"anat/sub-001_T1w.nii"；
3. 选中"Save Bias Corrected"，并选择"Save Bias Corrected"；
4. 选中"Deformation Fields"列表按钮并选择"Forward"。保存成segment.mat并运行。
5. 使用"Check Reg"工具查看生成的灰质"anat/c1sub-001_T1w.nii"和原始（对齐后的）功能像（第一张图像）"func/rsub-001_task-Training_run-01_bold.nii,1"
6. 查看（已经配准过的）结构像"anat/sub-001_T1w.nii"和偏置校正图像"anat/msub-001_T1w.nii"的对比。
7. 同时生成变形场文件"anat/y_sub-001_T1w.nii"，包含编码x、y和z坐标的3卷。由于结构和功能数据都是对齐的，所以变形场文件可以被用于对功能数据进行空间正则化。

运行任务，因为后面的“正则化”步骤需要用到“重对齐”所生成的 `r*.nii` 文件。否则后面你的运行无效。


### 4. 正则化
对 功能像 进行正则化：为了增强模型拟合的精度，对每个体素使用减去平均响应并缩放到单位方差 的方法 进行响应的正则化。选择"Normalise (Write)"工具。
1. 选中"Data"，选择"New Subject"；
2. 选中"Deformation Field"，并选择分割步骤中产生的变形场文件"anat/y_sub-001_T1w.nii"；
3. 选中"Images to Write"，并选中所有已再对齐的镜像"func/rsub-001_task-Training_run-01_bold.nii,410"；(1:410)
4. 在"Writing Options"选项中，确认"Voxel sizes"是[2 2 2]；
5. 保存成"normalise_functional.mat"，并运行；
6. 运行并在功能数据目录产生以"w"（write）开头的空间正则化文件（func/wrsub-001_task-Training_run-01_bold.nii）。

对（偏置校正后的）结构像进行正则化：如果想一个受试的功能激活施加到他自己的解剖（结构像）上，需要对他们（偏置校正后的）解剖图像 应用空间正则化参数。
1. 选择"Normalise (Write)，选中"Data"，选择"New Subject"；
2. 选中"Deformation Field"，选择分割步骤中产生的形变场文件"anat/y_sub-001_T1w.nii"；
3. 选中"Image to Write"，选择偏置校正后的结构像"anat/msub-001_T1w.nii"，按下"Done"。
4. 打开"Writing Options"，选择体素大小，并确认是默认值[2 2 2]（这对应原始镜像的分辨率）。
5. 保存到"job/normalise_structural.mat"，并运行。

这会生成以"w"开头的正则化结构像（anat/wmsub-001_T1w.nii）。

运行任务，为后面的“平滑”步骤提供 `wr*.nii` 文件。


### 5. 平滑
对正则化后的功能像进行平滑，按"Smooth"工具按钮。
1. 选择"Images to Smooth"，并选择正则化步骤中产生的所有空间正则化文件"func/wrsub-001_task-Training_run-01_bold.nii,410"。(1:410)
2. 选中"FWHM"，将[8 8 8]改成[6 6 6]（不确定），这会使用6毫米在每一个方向对数据进行平滑。
3. 保存成"smooth.mat"并运行。这一步会生成平滑后的镜像 `swrsub-001_task-Training_run-01_bold.nii`。
4. 使用Check Reg工具同时打开正则化后的功能像`func/wrsub-001_task-Training_run-01_bold.nii,1` 和平滑后的功能像 `func/swrsub-001_task-Training_run-01_bold.nii,1`。

`s` 表示平滑 smooth，`w` 表示正则化，`r` 表示 重对齐 realign。



### 移除低频漂移（未使用）
使用一个240-s窗口的中值滤波 来消除 低频漂移。


### 识别和注册（未使用）
使用FreeSurfer从解剖数据中识别出皮层表面，并使用功能数据的体素 将它们注册。




## 1. 模型定义
构建、查看、估计模型
按下"Specify 1st-level"按钮。
1. 打开"Timing parameters"选项；
2. 选中"Units for design"并选择"Scan"；？
3. 选中"Interscan interval"并输入1.5，这是以秒为单位的TR；
4. 选中"Data and Design"并选择"New Subject/Session"，这会打开新创建的"Subject/Session"会话；
5. 选中"Scans"，并选择（不能直接选，需要用Filter和Frames过滤出来）410个平滑并正则化后的功能像"func/swrsub-001_task-Training_run-01_bold.nii"；
Filter填写"^swrsub-001_task-Training_run-01_bold.nii"，Frames填"1:410"回车，不然会报错："Not enough scans in session 1"。
[参考](https://en.wikibooks.org/wiki/SPM/Working_with_4D_data)
6. 选中"Condition"并选择"New condition"；
7. 打开新建的"Condition"选项；选中"Name"并输入"listening"。选中"Onsets"并输入"0:10:400"；选中"Durations"（事件的持续时间）并输入"10"。（每个音频段都是15s，对应10个volumn）。
8. 选中"Directory"，并选择创建好的目录"classical"。
9. 保存任务到"specify.mat"，并运行。SPM会写一个SPM.mat文件到classical目录（应该是matlab所在的当前目录），同时绘制了设计矩阵。

点击绿色右箭头（或者用保存的脚本`specify.m`运行 ）。

## 2. 模型估计
1. 按下"Estimation"按钮，选中"Select SPM.mat"选项，然后选出保存在"classical"子目录中的"SPM.mat"文件。
2. 将任务保存成"estimate.mat"，并运行。

这一步会在所选择的目录中写下许多文件（包括SPM.mat文件）。

用脚本运行 `estimate.m`。

估计生成的`classical`文件夹：
* beta_000k.img, 估计的回归系数镜像，k 表示第 k 个回归系数。
* ResMS.img, 误差方差的镜像。
* RPV.img, the estimated resels per voxel(每个像素的分解元件)。
* `con_000i` 和 `con_sd`，第 i 个预定义对比的均值和标准差（如果对比有指定的 SPM）。加权参数估计。


## 3. 模型推断
模型估计（生成了SPM.mat）之后，按下"Results"按钮，选择上一步生成的`SPM.mat`文件，这会调出`对比管理器`。

### 对比管理器
在对比管理器的右边的画板中显示设计矩阵（可交互的），并在左边画板中指定需要对比的。
可以选择`t contrasts`（或者"F对比"）。
为了检验条件下的效应的统计结果：
1. 选择`Define new contrast`。
2. 一边可以指定主要正在听的条件下的效果（比如：单边 t检验），（submit一次）在contras中的`contrast weights vector` 中输入 1 、（名字中输入）`listening > rest`； 
   （-1表示：`rest > listening`）。
   SPM仅仅能接受可以估计的对比。可以接受的对比在对比管理器中会以绿色显示，不正确的对比会以红色显示。
3. 选择对比名，比如：`listening > rest`。
4. 按下"Done"按钮。

### 掩模
1. "apply masking"选择"none"。

### 门限
1. 选择"FWE"；
2. "p value (FWE)" 接受默认0.05（直接回车）；
3. "& extent threshold {voxels}" 接受默认值0（直接回车）；

### 输出文件
这时会在工作目录(classical)生成很多文件。
包含 加权参数估计的镜像 保存为 `con_0001.nii`，T统计量的镜像保存为`spmT_0001.nii`和`spmT_0002.nii`。

### 最大密度投影

### 设计矩阵

### 统计表格

### 在体素上绘制响应

某个体素能够在（结果的）交互式窗口中根据对应的坐标进行选择。在这个体素的响应能够使用在交互式窗口可视化部分的`Plot`按钮进行绘制。SPM 提供了五种进一步的选项：
1. Contrast estimates and 90% CI：SPM会推出特定的对比（比方说选择listening>rest，表示听大于静息)；
2. Fitted responses（拟合的响应）：在回话/受试之间绘制调整后的数据和拟合好（选择adjusted；"plot against"选择"scan or time"）的响应。SPM会推出特定的对比并提供选项来选择不同的纵坐标（提供的选项包括一个可解释的变量、扫描或者时间、用户自己制定）；
3. Event-related responses（事件相关的响应）：选择fitted response and PSTH；
4. Parametric responses
5. Volterra kernels

### 叠加
交互式可视化窗口也为激活集群解剖可视化提供了叠加工具。按下"Overlays"会显示几个选项的下拉菜单：
1. Slice：覆盖在 三个邻近的（2毫米）横断面切片上。选择`classical/spmT_0001.nii`，但是不够清晰；
2. Sections：覆盖在三个交叉 （矢状、冠状、轴面的（横截面？）） 切片上；选择之前生成的 以"w"开头的正则化结构像`anat/wmsub-001_T1w.nii`；
3. 渲染：叠加到一卷被渲染的大脑上。

对于`Render`选项，首先为这个受试创建一个渲染文件，实现如下：
1. 使用`Normalise (Write)`工具，在两个镜像`anat/c1sub-001_T1w.nii`（分割生成的灰质）和`anat/c2sub-001_T1w.nii`，使用形变场文件`anat/y_sub-001_T1w.nii`，以及体素大小为[1 1 1]；
（这一步会生成 正则化(w)的灰质图像`anat/wc1sub-001_T1w.nii`和`anat/wc2sub-001_T1w.nii`）；
2. 从主菜单`Render`下拉菜单中选择`Extract Surface`；
3. 选择正则化分割后的灰质，即第一步生成的两个白质文件`wc1sub-001_T1w.nii`和`wc2sub-001_T1w.nii`；
4. 使用默认的选项保存结果(Rendering and Surface)，
这一步 SPM 在图形窗口中渲染解剖镜像，保存为`anat/render_wc1sub-001_T1w.mat`（但是没看到激活）。
表面镜像没找到？

也可以在表面 mesh 上投影和显示结果。
这里使用经典的在 SPM 中发布的 mesh。
按下`Overlays`，并选择`Render`，到 SPM 安装目录中的`canonical`文件夹中选择`cortex_20484.surf.gii`）。



## 问题

#### This model has not been estimated

A: 模型估计（生成了SPM.mat）之后，按下"Results"按钮，选择上一步生成的"SPM.mat"文件，出现的问题。

#### The images do not all have same orientation and/or voxel sizes

A: 重新对齐（头动校正了，但是没有rescliced）。

#### The images do not all have same orientation and/or voxel sizes.
A: 猜测是，这反映了早期尝试对 TPM.nii 文件中的第一张图像进行配准或重新对齐（这导致 TPM.nii 文件中的 6 个组织图未配准）。 如果我的猜测是正确的，spm12/tpm 文件夹中会有一个名为 TPM.mat 的文件，只需删除此文件即可解决此问题。
[参考](https://www.nitrc.org/forum/forum.php?thread_id=8290&forum_id=1144)

A: If you’re working in MNI space, then you should be able to specify --output-space MNI152NLin2009cAsym:res-2, and the voxel sizes will be normalized to 2mm isotropic. 
[参考](https://neurostars.org/t/differences-in-orientation-voxel-size-pre-and-post-scanner-upgrade/4944)

### File "/data3/dong/brain/auditory/music_genre_fMRI/preprocess/swrsub-001_task-Test_run-01_bold.nii" does not exist.
A: spm_select选择的只是文件名，需要用 ExtFPListRec 来选择文件所在的完整路径




# 参考
* [人类大脑中音乐类别对应物和基于特征的表征](https://doi.org/10.1002/brb3.1936)

/data2/whd/win10/doc/paper/doctor/doctor.Data/PDF/1580343150

* SPM12文档

/data2/whd/win10/learn/neuro/research/SPM_manual.pdf
 