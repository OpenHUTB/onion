#!/usr/bin/env python
# coding: utf-8
# 原来是：brain-score/candidate_models/examples/model-activations.py

# ## 对于单张图像的激活
# 首先加载图像并显示它。
import numpy as np
from PIL import Image
from matplotlib import pyplot

stimuli_path = '/data3/dong/data/brain/visual/cornet/candidate_models-master/examples/image.jpg'
pyplot.imshow(np.array(Image.open(stimuli_path)))
pyplot.show()


# 对于预定义的模型，在 `candidate_models.models.implementations.model_layers` 中有一个默认深度网络层的选择。
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers

# 现在我们能
# （1）使用预定义的模型；
model = base_model_pool['resnet-101_v2']  # (1)
# （2）指定我们想从中抽取特征的深度网络层；
layers = model_layers['resnet-101_v2'][-2:]  # (2)
# （3）从模型中获得激活。
activations = model(stimuli=[stimuli_path], layers=layers)  # (3)

# `activations` 现在是有两个 `stimulus_path` 和 `neuroid` 两个维度的 xarray DataArray，
# 在 `neuroid` 中也有额外的元数据 `layer`。
print("\n", activations)


# ### 自定义的 Pytorch 模型
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


# 现在你可以给 Pytorch wrapper 传递一个带有预处理函数的模型
import functools
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images

preprocessing = functools.partial(load_preprocess_images, image_size=224)
wrapper = PytorchWrapper(identifier='my-model', model=MyModel(), preprocessing=preprocessing)

# 现在你可以对指定层的列表抽取激活
activations = wrapper(stimuli=[stimuli_path], layers=['linear', 'relu2'])
print(activations)

# ### 自定义 Tensorflow 模型
from model_tools.activations.tensorflow import TensorflowSlimWrapper, load_resize_image
import tensorflow as tf

slim = tf.contrib.slim
tf.reset_default_graph()

image_size = 224
placeholder = tf.placeholder(dtype=tf.string, shape=[64])
preprocess = lambda image_path: load_resize_image(image_path, image_size)
preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)

with tf.variable_scope('my_model', values=[preprocess]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
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
wrapper = TensorflowSlimWrapper(identifier='tf-custom', labels_offset=0, 
                                endpoints=endpoints, inputs=placeholder, session=session)
activations = wrapper(stimuli=[stimuli_path], layers=['my_model/pool2'])
print(activations)

