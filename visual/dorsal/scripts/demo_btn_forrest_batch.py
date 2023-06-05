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

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize  # ImportError: cannot import name imread  -> install pillow

import pandas as pd


from hart.data import disp
from hart.data.kitti.tools import get_data
from hart.model import util
from hart.model.attention_ops import FixedStdAttention
from hart.model.eval_tools import log_norm, log_ratios, log_values, make_expr_logger
from hart.model.tracker import HierarchicalAttentiveRecurrentTracker as HART
from hart.model.nn import AlexNetModel, IsTrainingLayer
from hart.train_tools import TrainSchedule, minimize_clipped

from utils import tools

import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

batch_size = 1
img_size = 187, 621, 3
raw_img_size = 720, 1280, 3
crop_size = 56, 56, 3

vert_scal = (img_size[0] * 1.0)/raw_img_size[0]
horo_scal = (img_size[1] * 1.0) / raw_img_size[1]

rnn_units = 100
norm = 'batch'
keep_prob = .75

img_size, crop_size = [np.asarray(i) for i in (img_size, crop_size)]
keys = ['img', 'bbox', 'presence']

bbox_shape = (1, 1, 4)

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
# checkpoint_path = 'checkpoints/kitti/pretrained/2017_07_06_16.41/model.ckpt-142320'
checkpoint_path = os.path.join(proj_dir,
                               'checkpoints/kitti/pretrained/model.ckpt-347346')  # '/data2/whd/workspace/sot/hart/checkpoints/kitti/pretrained/model.ckpt-347346'


# %% Load trained model
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

# 先将模型结构加载到计算图（简单来说就是需要先定义变量），然后创建saver对象;
saver = tf.train.Saver()  # ValueError: No variables to save
#with tf.Session() as sess:
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# # NotFoundError: Restoring from checkpoint failed.
# This is most likely due to a Variable name or other graph key that is missing from the checkpoint.
# Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
# Key IsTrainingLayer1/is_training_IsTrainingLayer1 not found in checkpoint
saver.restore(sess, checkpoint_path)
model.test_mode(sess)


# %% SOT
seg_dir = os.path.join(proj_dir, 'data', 'seg')
subs = os.listdir(seg_dir)
all_iou = 0.0
for s in range(len(subs)):
    img_dir = os.path.join(seg_dir, subs[s])
    seq_id = int(os.path.basename(img_dir))

    #img_dir = os.path.join(proj_dir, 'data/seg/5/')  # '/data2/whd/workspace/sot/hart/data/seg6/1/'
    imgs = load_imgs(img_dir)
    # bbox = [50, 216, 120, 47]  # [垂直, 水平, 高, 宽]
    gt_bbox = pd.read_csv(os.path.join(img_dir, 'gt.txt'), header=None)
    first_bbox = gt_bbox.loc[0:0]
    # bbox = list(bbox)
    bbox = first_bbox.values.tolist()
    bbox = bbox[0]
    tmp = bbox[1]; bbox[1] = bbox[0]; bbox[0] = tmp  # gt中的起始xy和hart相反
    # tmp = bbox[2]; bbox[2] = bbox[3]; bbox[2] = tmp  # gt中的起始wh和hart相反
    bbox[0] = int(bbox[0]) * vert_scal  # 缩放到深度模型适合的输入大小: 187, 621, 3
    bbox[1] = int(bbox[1]) * horo_scal
    bbox[2] = int(float(bbox[2])) * horo_scal  # 宽度缩放
    bbox[3] = int(float(bbox[3])) * vert_scal  # 高度缩放

    feed_dict = {x: imgs, y0: np.reshape(bbox, bbox_shape)}
    tensors = [model.pred_bbox, model.att_pred_bbox, model.glimpse, model.obj_mask]
    pred_bbox, pred_att, glimpse, obj_mask = sess.run(tensors, feed_dict)

    feature_dir = os.path.join(img_dir, 'feature')
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    # sess.run([model.cell.features], feed_dict)[0]
    # [shape_first_att(1*4), shape_first_presence(1*1), zero_state(1*100,1*100), rnn_outputs(1*100)
    # 第一帧用于初始化，提取的特征为2到n的特征？
    np.savetxt(os.path.join(feature_dir, "frame-1-%02d_vt_feature" % (len(imgs))),
               np.squeeze( sess.run([model.states_flat], feed_dict)[0] ))  # 9*110 [s_t(n*100) * \upsilon_t(n*10) -> s_t(n*110)]
    # np.savetxt(os.path.join(feature_dir, "frame-1-%02d_vt_feature" % (len(imgs))),
    #           np.squeeze(sess.run([model.cell.features], feed_dict)[0]))  # 1*100
    np.savetxt(os.path.join(feature_dir, "frame-1-%02d_LSTM_feature" % (len(imgs))),
               np.squeeze( sess.run([model.rnn_output], feed_dict)[0][1:] ))     # 10*100 去除第一个rnn的输出，和model.states_flat的长度保持一致
    # np.savetxt('result/activation/MLP', sess.run([model.hidden_to_bbox.output], feed_dict)[0])
    # [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # sess.graph.get_tensor_by_name(u'HierarchicalAttentiveRecurrentTracker/AttentionCell/to_features:0')
    # sess.graph.get_tensor_by_name()

    # %% 保存跟踪结果
    n = imgs.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(20, 2 * n))  # (row,column): (n, 3)

    # 保存跟踪结果
    saved_dir = os.path.join(img_dir, 'track')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    saved_pred_bbox = pred_bbox.squeeze().copy()
    # 将跟踪结果框缩放到原图尺寸。 [垂直, 水平, 高, 宽]
    saved_pred_bbox[:, 0] = saved_pred_bbox[:, 0] / vert_scal
    saved_pred_bbox[:, 1] = saved_pred_bbox[:, 1] / horo_scal
    saved_pred_bbox[:, 2] = saved_pred_bbox[:, 2] / horo_scal
    saved_pred_bbox[:, 3] = saved_pred_bbox[:, 3] / vert_scal
    np.savetxt(os.path.join(saved_dir, 'pred_bbox.txt'), saved_pred_bbox)

    #%% 计算跟踪的IOU
    saved_pred_bbox, gt_bbox
    gt_bbox = gt_bbox.values
    cur_iou = 0.0
    print('Sequence ID: %d' % seq_id)
    for i in range(len(gt_bbox)):
        cur_pred_bbox = saved_pred_bbox[i, :]
        cur_gt_bbox = gt_bbox[i, :]
        tmp = cur_pred_bbox[0]; cur_pred_bbox[0] = cur_pred_bbox[1]; cur_pred_bbox[1] = tmp  # 起始点的垂直水平交换
        # 宽高转换为坐标
        cur_pred_bbox[2] = cur_pred_bbox[0] + cur_pred_bbox[2]
        cur_pred_bbox[3] = cur_pred_bbox[1] + cur_pred_bbox[3]
        cur_gt_bbox[2] = cur_gt_bbox[0] + cur_gt_bbox[2]
        cur_gt_bbox[3] = cur_gt_bbox[1] + cur_gt_bbox[3]
        iou_tmp = tools.compute_iou(cur_pred_bbox, cur_gt_bbox)
        cur_iou += iou_tmp
        print(iou_tmp)
    all_iou += cur_iou / len(gt_bbox)
    print('********************************************')

    #%% 绘制跟踪结果图并保存
    gt_color = 'r'
    pred_color = 'b'
    for i, ax in enumerate(axes):
        # ax[0].imshow(imresize(imgs[i].squeeze(), raw_img_size[:2]) / 255.)  # 187*621*3
        ax[0].imshow(imgs[i].squeeze() / 255.)
        ax[1].imshow(glimpse[i].squeeze())  # step 1: 56*56*3
        ax[2].imshow(obj_mask[i].squeeze(), cmap='gray', vmin=0., vmax=1.)  # step 2: 14*14
        disp.rect(pred_bbox[i].squeeze(), gt_color, ax=ax[0])   # 预测的框 是 gt？: 4
        disp.rect(pred_att[i].squeeze(), pred_color, ax=ax[0])  # 预测的注意力 : 4
        for a in ax:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)

        axes[i, 0].plot([], c=gt_color, label='gt')  # ground truth, red
        axes[i, 0].plot([], c=pred_color, label='pred')  # prediction result, blue
        axes[i, 0].legend(loc='center right')
        axes[i, 0].set_xlim([0, img_size[1]])
        axes[i, 0].set_ylim([img_size[0], 0])

    track_res_dir = os.path.join('result', 'track')
    if not os.path.exists(track_res_dir):
        os.mkdir(track_res_dir)
    plt.savefig(os.path.join(track_res_dir, 'track_forrest_%d.png' % seq_id ))
    pass

all_iou /= len(subs)
print('All IoU: %f\n' % all_iou)  # All IoU: 0.040300


#%% TODO
# 需要提高跟踪性能的序列：2,6,10
# 很难的序列：5,12


