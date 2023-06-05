# -*- coding:utf-8 -*-
# 中文注释报错：SyntaxError: Non-ASCII character '\xe8'
# %%
# --img_dir=/data2/whd/workspace/sot/data/hart/training/image_02/ --label_dir=/data2/whd/workspace/sot/data/hart/training/label_02/ --is_debug True

import matplotlib
matplotlib.use('TkAgg')  # TkAgg, Agg

import os

import sys
# pycharm运行py文件会自动将当前运行py文件的工程目录加载到sys.path中，不加这个命令行运行会报错：ImportError: No module named hart.data
sys.path.append("/data2/whd/workspace/sot/hart")
script_dir = os.path.split(os.path.realpath(__file__))[0]
proj_dir = os.path.split(script_dir)[0]

os.chdir('../')  # 加上这个命令行运行报错？ IOError: [Errno 2] No such file or directory: 'result/activation/LSTM'

# print(os.getcwd())
# sys.path.append('neurocity')
# sys.path.append('neurocity/component')

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize  # ImportError: cannot import name imread  -> install pillow


from hart.data import disp
from hart.data.kitti.tools import get_data
from hart.model import util
from hart.model.attention_ops import FixedStdAttention
from hart.model.eval_tools import log_norm, log_ratios, log_values, make_expr_logger
from hart.model.tracker import HierarchicalAttentiveRecurrentTracker as HART
from hart.model.nn import AlexNetModel, IsTrainingLayer
from hart.train_tools import TrainSchedule, minimize_clipped


import matplotlib.pyplot as plt
# % matplotlib
# inline

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# %%

def load_imgs(img_dir):
    imgs_list = []
    for file in os.listdir(img_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            imgs_list.append(file)
    img_paths = sorted(imgs_list)
    imgs = np.empty([len(img_paths), 1] + list(img_size), dtype=np.float32)
    for i, img_path in enumerate(img_paths):
        img_path = os.path.join(img_dir, img_path)
        imgs[i, 0] = imresize(imread(img_path, mode="RGB"), img_size[:2])

    return imgs


# %%
alexnet_dir = os.path.join(proj_dir, 'checkpoints')  # '/data2/whd/workspace/sot/hart/checkpoints'
img_dir = os.path.join(proj_dir, 'data/seg/5/')  #  '/data2/whd/workspace/sot/hart/data/seg6/1/'
# checkpoint_path = 'checkpoints/kitti/pretrained/2017_07_06_16.41/model.ckpt-142320'
checkpoint_path = os.path.join(proj_dir, 'checkpoints/kitti/pretrained/model.ckpt-347346') #  '/data2/whd/workspace/sot/hart/checkpoints/kitti/pretrained/model.ckpt-347346'

batch_size = 1
img_size = 187, 621, 3
crop_size = 56, 56, 3

rnn_units = 100
norm = 'batch'
keep_prob = .75

img_size, crop_size = [np.asarray(i) for i in (img_size, crop_size)]
keys = ['img', 'bbox', 'presence']

bbox_shape = (1, 1, 4)

# %%

tf.reset_default_graph()
util.set_random_seed(0)

x = tf.placeholder(tf.float32, [None, batch_size] + list(img_size), name='image')
y0 = tf.placeholder(tf.float32, bbox_shape, name='bbox')
p0 = tf.ones(y0.get_shape()[:-1], dtype=tf.uint8, name='presence')

is_training = IsTrainingLayer()
builder = AlexNetModel(alexnet_dir, layer='conv3', n_out_feature_maps=5, upsample=False, normlayer=norm,
                       keep_prob=keep_prob, is_training=is_training)

model = HART(x, y0, p0, batch_size, crop_size, builder, rnn_units,
             bbox_gain=[-4.78, -1.8, -3., -1.8],
             zoneout_prob=(.05, .05),
             normalize_glimpse=True,
             attention_module=FixedStdAttention,
             debug=True,
             transform_init_features=True,
             transform_init_state=True,
             dfn_readout=True,
             feature_shape=(14, 14),
             is_training=is_training)

# %%
saver = tf.train.Saver()
with tf.Session() as sess:
    # %% Plot network structure
    # /home/d/anaconda2/envs/hart/bin/tensorboard --logdir=/data2/whd/workspace/sot/hart/result/log/tensorflow
    summary_writer = tf.summary.FileWriter('result/log/tensorflow/', sess.graph)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path)
    model.test_mode(sess)

    imgs = load_imgs(img_dir)
    bbox = [50, 216, 120, 47]  # [x0, y0, height, width]

    feed_dict = {x: imgs, y0: np.reshape(bbox, bbox_shape)}
    tensors = [model.pred_bbox, model.att_pred_bbox, model.glimpse, model.obj_mask]
    pred_bbox, pred_att, glimpse, obj_mask = sess.run(tensors, feed_dict)
    # xxx np.savetxt('result/activation/V1', np.squeeze(sess.run([model.cell.before_features], feed_dict)[0]))
    # np.savetxt('result/activation/vt', np.squeeze(sess.run([model.cell.features], feed_dict)[0]))
    np.savetxt(os.path.join(proj_dir, 'result/activation/LSTM'), np.squeeze(sess.run([model.rnn_output], feed_dict)[0]))
    # np.savetxt('result/activation/MLP', sess.run([model.hidden_to_bbox.output], feed_dict)[0])
    pass


# %% Plot tracking results
from matplotlib.animation import FFMpegFileWriter

# metadata = dict(title='tracking demo',
#                 artist='Matplotlib',
#                 comment='draw tracking demo')
# writer = FFMpegFileWriter(fps=15, metadata=metadata)

n = imgs.shape[0]
fig, axes = plt.subplots(n, 3, figsize=(20, 2 * n))  # (row,column): (n, 3)
# fig = plt.figure()
# plt.show()

# ../result/
# with writer.saving(fig, 'tracking_demo.mp4', 100):
gt_color = 'r'
pred_color = 'b'
for i, ax in enumerate(axes):
    ax[0].imshow(imgs[i].squeeze() / 255.)
    ax[1].imshow(glimpse[i].squeeze())  # step 1
    ax[2].imshow(obj_mask[i].squeeze(), cmap='gray', vmin=0., vmax=1.)  # step 2
    disp.rect(pred_bbox[i].squeeze(), gt_color, ax=ax[0])
    disp.rect(pred_att[i].squeeze(), pred_color, ax=ax[0])
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)

    axes[i, 0].plot([], c=gt_color, label='gt')   # ground truth, green
    axes[i, 0].plot([], c=pred_color, label='pred')  # prediction result, blue
    axes[i, 0].legend(loc='center right')
    axes[i, 0].set_xlim([0, img_size[1]])
    axes[i, 0].set_ylim([img_size[0], 0])

    # plt.savefig("result/%d.png" % i)
        # writer.grab_frame()

plt.savefig("result/final_forrest.png")

