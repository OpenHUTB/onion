# 参考brain/visual/metrics/demo.py
# 实现类脑音频相似性分数

# 相似性度量
# 该框架关注于两个数据集的比较，比如源脑（深度网络的激活）和目标脑（大脑皮层的激活）。
# 它不负责源数据的多种组合（例如模型中的多个层），而只进行直接比较。

# 该度量标准表示了数据集之间的相似性，
# 为了进行比较，他们可能会被重新映射（神经预测性）或者在一个子空间（RDMs）中进行比较。

# #### 具有皮尔逊相关性的神经预测
import os.path

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


# 加载fMRI的 T2/T3 区域的激活数据
abs_path = __file__  # 获得当前文件的绝对路径（包括文件名）
# 从右向左边查询
# pro_dir = abs_path[:abs_path.rfind("/")]  # windows下用\\分隔路径，linux下用/分隔路径
pro_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

fMRI_intensity_dir = os.path.join(pro_dir, "data/brain/auditory/ds003720-download/sub-001/intensity")

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
print(score.data[0])




