#!/usr/bin/env python
# coding: utf-8

# 包括Brain-Score/example/data.py、metrics.py、beanchmarks.py 三个脚本文件的内容

# 安装：
# brainio-base==0.1.0
# brainio-collection==0.1.0

# /data3/dong/data/brain/visual/cornet/brain-score-0.2
# /home/d/anaconda2/envs/brain_model/bin/python ./setup.py install
# /home/d/anaconda2/envs/brain_model_3_10/bin/python ./setup.py install


# BrainScore 基准包括一个目标数据集（测量的大脑激活）和一个进行数据比较的度量标准。
# 该基准接受一个源数据集，来进行比较并产生一个相似性分数。

# 预定义的基准
# Brainscore 定义了一个能够在类脑模型上运行的基准。
# 将被评测的类脑模型实现了 BrainModel 接口。
# 一个非常简单的实现如下所示：

import numpy as np
from typing import List, Tuple
# from brainscore.benchmarks.screen import place_on_screen

from brainscore.model_interface import BrainModel
from brainio.assemblies import DataAssembly

import brainscore
from brainscore import score_model

from brainscore.benchmarks.screen import place_on_screen


# 这块代码主要关注原始数据的加载，这些数据通常没有经过充分的预处理
# （比如，没有过滤掉不信任的神经元，重复没有被平均，硬刺激没有被预先选择等等）
# (e.g. neuroids that we don't trust are not filtered, repetitions are not averaged, hard stimuli are not pre-selected etc.).
#
# 如果你仅仅想进行数据之间的互相比较，最好直接使用基准
# （比如：`from brainscore import benchmarks; benchmarks.load('dicarlo.MajajHong2015')`）
# 或者通过基准加载数据
# （比如：`from brainscore import benchmarks; benchmarks.load_assembly('dicarlo.MajajHong2015')`）

# ### 神经记录的集合 Neural assembly
# 我们可以使用 `get_assembly` 方法加载数据（叫做 "assembly" ）
# 在下面的试验中，从 DiCarlo 实验室公布在论文 Majaj, Hong et al. 2015 中加载神经数据。
neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")  # 256*148480*1
# 这会得到一个 NeuronRecordingAssembly 对象（是 xarray DataArray 的子类）。
# 行为和神经集合通常使用 xarray 框架进行处理。
# xarray 数据是一个类似于 pandas 的基本的带有注解坐标的多维表格。
# 更多的信息可以参考: http://xarray.pydata.org
#
# 将神经记录集合 `dicarlo.MajajHong2015.public` 结构化为维度 `神经元 x 表征`
# `neuroid` 是一个 包含记录位置信息的多索引结构 MultiIndex，比如动物和脑区。
# `presentation` 指带有注释坐标（比如图片id `image_id` 和 repetition）的单个表征刺激。
# 最后，`time_bin` 告诉我们从收集神经响应开始的时间（以毫秒为单位）。
# 这个神经记录集合包含 70-170 毫秒窗口中的平均峰值速率。

# 这些数据是未经过加工过的，但是一般我们使用一个经过预处理后的版本。
# 我们按照下面的步骤对数据进行处理：
# 1. 对横跨各个 repetitions 的数据进行平均； 256*148480*1 -> 256*3200*1
compact_data = neural_data.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')
# 2. 从下颞区进行神经元的过滤； -> 168*3200*1
compact_data = compact_data.sel(region='IT')
# 3. 去除标量 time_bin 的维度； -> 168*3200
compact_data = compact_data.squeeze('time_bin')
# 4. 并改变维度成 `presentation x neuroid`； -> 3200*168
# 现在数据包含 3200 张图像和 168 个神经元的响应
compact_data = compact_data.transpose('presentation', 'neuroid')

# 注意在基准测试中使用的数据通常是已经经过预处理。
# 例如，公共基准 `MajajITPublicBenchmark` 的目标集（皮层记录）和我们前面四步预处理的结果一样：
from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark

benchmark = MajajHongITPublicBenchmark()
benchmark_assembly = benchmark._assembly  # 3200*168

# ### 刺激集
# 你可能注意到之前集合中的 `stimulus_set` 属性。
# 刺激集包含用于测量神经记录的刺激。
# 具体来说，这需要例如 image_id 和 object_name 等信息，打包在 pandas DataFrame 中。
# Specifically, this entails e.g. the image_id and the object_name, packaged in a pandas DataFrame.
stimulus_set = neural_data.attrs['stimulus_set']  # 3200*18

# 可以使用 `get_image` 方法直接获取任意图片
image_path = stimulus_set.get_image(stimulus_set['image_id'][0])

# 图片自动下载到本地，并能够直接加载和显示
from matplotlib import pyplot, image
img = image.imread(image_path)
# pyplot.imshow(img)
# pyplot.show()


##*********************************************************************************************************************
# 相似性度量
# 该框架关注于两个数据集的比较，比如源脑（深度网络的激活）和目标脑（大脑皮层的激活）。
# 它不负责源数据的多种组合（例如模型中的多个层），而只进行直接比较。

# 该度量标准表示了数据集之间的相似性，
# 为了进行比较，他们可能会被重新映射（神经预测性）或者在一个子空间（RDMs）中进行比较。

# #### 具有皮尔逊相关性的神经预测
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation

# ### 预定义的度量标准
# Brain-Score 附带了该领域使用的许多标准指标。
# 一个标准的度量标准是神经预测性：
# （1）在两个系统之间使用线性回归，来进行线性映射（比如从神经激活到神经放电率）；
regression = pls_regression()  # 1: 定义回归
# （2）它计算在保留（hold-out）图像上所预测的放电率之间的相关性；
correlation = pearsonr_correlation()  # 2: 定义相关性
# （3）并将所有这些都包含在交叉验证中以估计泛化能力。
metric = CrossRegressedCorrelation(regression, correlation)  # 3: 包含在交叉验证中

# 在一些数据集中运行相似性度量来获得分数：
import numpy as np
from numpy.random import RandomState

from brainio.assemblies import NeuroidAssembly

rnd = RandomState(0)  # 为了复现
# 随机生成一些皮层激活记录的数据
assembly = NeuroidAssembly((np.arange(30 * 25) + rnd.standard_normal(30 * 25)).reshape((30, 25)),  # 30个，每个25维
                           coords={'image_id': ('presentation', np.arange(30)),  # 30张图像，id依次变大
                                   'object_name': ('presentation', ['a', 'b', 'c'] * 10),  # 图像类别
                                   'neuroid_id': ('neuroid', np.arange(25)),  # 激活位置的id
                                   'region': ('neuroid', [0] * 25)},          # 激活位置的值
                           dims=['presentation', 'neuroid'])  # 30个图像对应30个表征presentation，每个对应25个神经元neuroid 
prediction, target = assembly, assembly  # 测试相似性度量是否能够预测和它们自己之间的相似性
score = metric(source=prediction, target=target)  # 2*1：第一维表示有多相似

# 上面的分数值是 分割和神经元 的聚合。
# 也可以检查原始值，比如每个分割和每个神经元的值。
# print(score.raw)


# #### RDM
# Brain-Score 也包括不需要任何拟合的比较方法，比如表征差异矩阵（Representational Dissimilarity Matrix，RDM）
from brainscore.metrics.rdm import RDMCrossValidated

metric = RDMCrossValidated()
rdm_score = metric(assembly1=assembly, assembly2=assembly)

# ### 自定义度量标准
# 一个简单地返回两个集合相似性分数的度量标准。
# 例如，下面的代码计算了回归的欧式距离和目标神经元。
# For instance, the following computes the Euclidean distance of regressed and target neuroids.
from brainio.assemblies import DataAssembly
from brainscore.metrics.transformations import CrossValidation
from brainscore.metrics.xarray_utils import XarrayRegression
from brainscore.metrics.regression import LinearRegression


class DistanceMetric:
    def __init__(self):
        regression = LinearRegression()
        self._regression = XarrayRegression(regression=regression)
        self._cross_validation = CrossValidation()

    def __call__(self, source, target):
        return self._cross_validation(source, target, apply=self._apply, aggregate=self._aggregate)

    def _apply(self, source_train, target_train, source_test, target_test):
        self._regression.fit(source_train, target_train)
        prediction = self._regression.predict(source_test)
        score = self._compare(prediction, target_test)
        return score

    def _compare(self, prediction, target):
        prediction, target = prediction.sortby('image_id').sortby('neuroid_id'), target.sortby('image_id').sortby(
            'neuroid_id')
        assert all(prediction['image_id'].values == target['image_id'].values)
        assert all(prediction['neuroid_id'].values == target['neuroid_id'].values)
        difference = np.abs(target.values - prediction.values)  # lower is better
        return DataAssembly(difference, coords=target.coords, dims=target.dims)

    def _aggregate(self, scores):
        return scores.median('neuroid').mean('presentation')


metric = DistanceMetric()
score = metric(assembly, assembly)
print(score)


##*********************************************************************************************************************
# 实现给定脑区到神经网络层的映射。
# 在 look_at 方法中，该类仅创建并返回一个模拟的结果。
# 其他两个方法仅仅检查输入值的正确性。

# 这个将来被测试类脑相似性的模型实现了 BrainModel 接口
class RandomITModel(BrainModel):
    def __init__(self):
        self._num_neurons = 50
        # 注意记录了哪些时间
        self._time_bin_start = None
        self._time_bin_end = None

    def look_at(self, stimuli, **kwargs):
        print(f"Looking at {len(stimuli)} stimuli")
        rnd = np.random.RandomState(0)
        recordings = DataAssembly(rnd.rand(len(stimuli), self._num_neurons, 1),
                              coords={'image_id': ('presentation', stimuli['image_id']),
                                      'object_name': ('presentation', stimuli['object_name']),
                                      'neuroid_id': ('neuroid', np.arange(self._num_neurons)),
                                      'region': ('neuroid', ['IT'] * self._num_neurons),
                                      'time_bin_start': ('time_bin', [self._time_bin_start]),
                                      'time_bin_end': ('time_bin', [self._time_bin_end])},
                              dims=['presentation', 'neuroid', 'time_bin'])
        recordings.name = 'random_it_model'
        return recordings

    def start_task(self, task, **kwargs):
        print(f"Starting task {task}")
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target=BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
        print(f"Recording from {recording_target} during {time_bins} ms")
        if str(recording_target) != "IT":
            raise NotImplementedError(f"RandomITModel only supports IT, not {recording_target}")
        if len(time_bins) != 1:
            raise NotImplementedError(f"RandomITModel only supports a single start-end time-bin, not {time_bins}")
        time_bins = time_bins[0].tolist()
        self._time_bin_start, self._time_bin_end = time_bins[0], time_bins[1]

    # 声明模型有 8 度的视场大小
    def visual_degrees(self):
        print("Declaring model to have a visual field size of 8 degrees")
        return 8

# 以下代码行加载了公共基准“MajajHong2015public.IT-pls”，
# 包括了从论文“Majaj, Hong et al. 2015” 中恒河猴下颞的神经记录，
# 和一个基于 PLS 回归的神经预测性度量，来进行模型预测和实际数据的比较。
# 返回带有 “RandomITModel” 对象的基准，然后返回在特定基准下模型的大脑相似性分数。

# IT 的大脑模型
model = RandomITModel()

score = score_model(model_identifier='mymodel', model=model, benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)

# 该基准
#（1）该基准记录了模型对 2560 个刺激得响应，
#（2）使用神经预测性度量比较所预测的模型响应和实际灵长类大脑皮层记录，并产生类脑相似性分数
#（3）通过向上取整对分数进行正则化。
# 因为基准已经交叉验证过结果，最后的分数包括中心（分割的平均值，这种情况下是平均值）和误差（这种情况下是平均的标准差）。
# the resulting score now contains the center (i.e. the average of the splits, in this case the mean)
# and the error (in this case standard-error-of-the-mean).

center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
print(f"score: {center.values:.3f}+-{error.values:.3f}")


# 该分数表示随机特征不能很好的预测下颞的记录。
# 也可以检查原始未经过向上取整的值。
# We can also check the raw unceiled values...
unceiled_scores = score.raw
print(unceiled_scores)

# ...as well as the per-neuroid, per-split correlations.
raw_scores = score.raw.raw
print(raw_scores)


# 自定义基准
# 也可以定义自己的基准，该基实现两个目的：
# 1. 在模型上复现灵长类实验；
# 2. 使用相似性度量来比较预测值和实际测量值；
# 3. 正则化匹配上线值，比如：模型性能的上线
# 3. normalize the match with the ceiling, i.e. an upper bound on how well a model could do
#
# 下面的例子实现了上面三步的一个简单的基准

import numpy as np
import brainscore
from brainscore.benchmarks import Benchmark
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.benchmarks._neural_common import explained_variance, average_repetition


# 我们想测试100-120毫秒之间模型预测和下颞记录的匹配程度
# 使用的是 Majaj et al. 2015 论文中从灵长类被动注视所采集而来的数据
class MyBenchmark(Benchmark):
    def __init__(self):
        # 刺激集StimulusSets和皮层记录集assemblies都通过 https://github.com/brain-score/brainio 进行打包
        assembly = brainscore.get_assembly(
            'dicarlo.MajajHong2015.temporal.public')  # 这一步会花费一些时间来进行数据的下载（5.79G）和打开
        assembly = assembly[{'time_bin': [start == 100 for start in assembly['time_bin_start'].values]}]
        # 查看一下图像数据集的一个子集（前1000个）
        image_ids = np.unique(assembly['image_id'].values)[:1000]
        assembly = assembly.loc[{'presentation': [image_id in image_ids for image_id in assembly['image_id'].values]}]
        stimulus_set = assembly.stimulus_set  # 皮层记录都有一个刺激集合StimulusSet和它绑定
        stimulus_set = stimulus_set[stimulus_set['image_id'].isin(image_ids)]
        assembly.attrs['stimulus_set'] = stimulus_set
        # 为了简化问题，减小 x 个神经元的表征唯独（仅仅有一个时间窗口time_bin）
        # reduce to presentation x neuroid for simplicity (we only have one time_bin)
        assembly = assembly.squeeze('time_bin')
        # 注意：这个皮层记录仍然会有重复（需要向上取整）
        self._assembly = assembly  # note that this assembly still has repetitions which we need for the ceiling
        self._similarity_metric = CrossRegressedCorrelation(
            regression=pls_regression(), correlation=pearsonr_correlation(),
            crossvalidation_kwargs=dict(splits=3, stratification_coord='object_name'))
        self._ceiler = InternalConsistency()

    @property
    def identifier(self):  # 为了进行结果的存储
        return "my-dummy-benchmark"

    def __call__(self, candidate: BrainModel):
        # 因为该候选模型 candidate 实现了 BrainModel 的接口，所以我们可以使用相同的方式来处理所有的模型
        # （1）在模型上复现实验
        candidate.start_task(task=BrainModel.Task.passive)
        candidate.start_recording(recording_target="IT", time_bins=[np.array((100, 120))])
        # 因为不同的模型会有不同的视野，所以我们调整了对应图片的大小
        # 例如，对于视野为 10 度的模型，2 度的刺激应该占用很少的空间，而对于 4 度的模型，相同的刺激将占用更多的空间。
        # for instance, a stimulus of 2 degree should take up little space for a model with a field of view of 10 degree
        # while the same stimulus would take up much more space for a model of 4 degrees.
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       # 作为参考，我们知道这个实验是在灵长类动物视野为 8 度下进行的
                                       source_visual_degrees=8)
        predictions = candidate.look_at(stimuli=stimulus_set)
        # （2）计算预测值和测量值之间的相似性
        assembly = average_repetition(self._assembly)  # 重复取平均
        predictions = predictions.squeeze('time_bin')
        print("Computing model-match")
        unceiled_score = self._similarity_metric(predictions, assembly)
        # （3）通过所估计的理想模型应该的表现来进行正则化
        ceiled_score = explained_variance(unceiled_score, self.ceiling)
        return ceiled_score

    @property
    def ceiling(self):
        print("Computing ceiling")
        return self._ceiler(self._assembly)


my_benchmark = MyBenchmark()
model = RandomITModel()  # 我们会使用和之前相同的模型
score = my_benchmark(model)
print(score)

# 我们也可以用自己的方法从原始数据从创建自定义的基准。
# 要与 Brain-Score 的其余部分进行交互，最简单的方法是将它们提供给 Benchmark 类。
# （但是我们不会自己继承和定义 `__call__` 方法）




