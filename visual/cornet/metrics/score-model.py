#!/usr/bin/env python
# coding: utf-8
# 原来是：brain-score/candidate_models/examples/score-model.py

# ## 自定义模型

# ### PyTorch
# 首先定义自己的模型
import numpy as np
import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
        self.linear = torch.nn.Linear(int(linear_input_size), 1000)
        self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu2(x)
        return x


# 可以定义自己的预处理过程，将文件路径转化为（正则化的）模型输入。
# 我们会使用标准定义的函数来将 224 x 224 大小的图像正则化到 ImageNet 的均值。
# 需要下载 `brain-score/model-tools-master` 并进行安装
# $ /home/d/anaconda2/envs/brain_model/bin/python ./setup.py install
import functools
from model_tools.activations.pytorch import load_preprocess_images

preprocessing = functools.partial(load_preprocess_images, image_size=224)

# 然后，使用 `PytorchWrapper`，将模型转换成激活模型。
# 激活模型对输入进行响应，并让我们能够从模型的任意层抽取激活
from model_tools.activations.pytorch import PytorchWrapper

activations_model = PytorchWrapper(identifier='my-model', model=MyModel(), preprocessing=preprocessing)


# 在 Brain-Score 中的候选模型必须遵循
# 模型接口 [model_interface](https://github.com/brain-score/brain-score/blob/master/brainscore/model_interface.py)
# 例如，这涉及决定哪些深度网络层映射到皮层区域。
# 如果想使用标注你的承诺（commitments），可以使用 `MomdelCommitment`。
# 为了将深度网络层映射到皮层区域，`ModelCommitment` 通过在公共基准上对所有层进行评分，凭经验确定最佳层。
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment

model = ModelCommitment(identifier='my-model', activations_model=activations_model,
                        # 指定需要考虑的层
                        layers=['conv1', 'relu1', 'relu2'])
# 打分模型 score_model 会在指定基准上对模型进行打分。
# 当要求模型从下颞区输出激活时（beanchmark_identifier指定为IT的激活数据），会首先搜索最佳深度网络层并输出这个最佳层的激活。
score = score_model(model_identifier=model.identifier, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)





"""
# ### Tensorflow (-Slim)
# TF-Slim是tensorflow中定义、训练和评估复杂模型的轻量级库。
# tf-slim中的组件可以轻易地和原生tensorflow框架以及例如tf.contrib.learn这样的框架进行整合。
# 首先，定义你的模型（其端点/终结点 endpoints）及其预处理功能。
from model_tools.activations.tensorflow import load_resize_image
import tensorflow as tf
slim = tf.contrib.slim
tf.reset_default_graph()

image_size = 224
placeholder = tf.placeholder(dtype=tf.string, shape=[64])
preprocess = lambda image_path: load_resize_image(image_path, image_size)
preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)  # 定义预处理功能

with tf.variable_scope('my_model', values=[preprocess]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # 收集 conv2d, fully_connected and max_pool2d 的输出
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
        net = slim.conv2d(preprocess, 64, [11, 11], 4, padding='VALID', scope='conv1')
        net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 1000, scope='logits')
        endpoints = slim.utils.convert_collection_to_dict(end_points_collection)

session = tf.Session()
session.run(tf.initialize_all_variables())



# 然后使用 `TensorflowSlimWrapper` 将自己的模型转换成激活模型。
# 激活模型队输入进行响应，并从模型的任意层抽取激活。
from model_tools.activations.tensorflow import TensorflowSlimWrapper

activations_model_tf = TensorflowSlimWrapper(identifier='tf-custom', labels_offset=0,
                                             endpoints=endpoints, inputs=placeholder, session=session)

# 在 Brain-Score 中的候选模型必须遵循
# 模型接口 [model_interface](https://github.com/brain-score/brain-score/blob/master/brainscore/model_interface.py)
# 例如，这涉及决定哪些深度网络层映射到皮层区域。
# 如果想使用标注你的承诺（commitments），可以使用 `MomdelCommitment`。
# 为了将深度网络层映射到皮层区域，`ModelCommitment` 通过在公共基准上对所有层进行评分，凭经验确定最佳层。
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment

model = ModelCommitment(identifier='tf-custom', activations_model=activations_model_tf,
                        # 指定需要考虑的深度网络层
                        layers=['my_model/conv1', 'my_model/pool1', 'my_model/pool2'])

score = score_model(model_identifier=model.identifier, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)


# 所有这些步骤都是示例实现，只要您的模型实现了模型接口 model_interface。

# ## 预定义模型
# 使用 `score_model` 方法和 `brain_translated_pool` 可以在一行中对基于神经数据的据模型进行评分。
# 模型预定义层会用来获取深度模型的激活。
# 就像模型实现一样，该方法调用的结果会缓存起来，以便它只需要计算一次。
from brainscore import score_model
from candidate_models.model_commitments import brain_translated_pool

identifier = 'alexnet'
model = brain_translated_pool[identifier]
score = score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)


# 分数通常伴随着对中心（比如平均值）和误差（比如平均值的标准误差）的估计。
# 这些值是对划分和神经元的聚合，并以基准上限为上限。
# These values are aggregations over splits and often neuroids, and ceiled by the benchmark ceiling.
# 
# 详细信息位于 https://github.com/brain-score/brain-score/blob/master/examples/benchmarks.ipynb
#
# 同时注意所有这些分数都在公开可得的数据上进行计算。
# 为了在完整的基准集上测试模型（包括保留的私有数据，即作者为了测试算法性能而保留的），请将解雇提交到：www.Brain-Score.org
# To test models on the full set of benchmarks (including held-out private data),
"""
