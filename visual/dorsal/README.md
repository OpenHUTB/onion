# 类脑视觉跟踪网络

* TODO
1. 计算眼动和单目标跟踪的相似度 （计算长方形的目标跟踪框和圆形的凝视范围之间的交并比）
2. 单目标跟踪、ground truth、眼动、平稳跟踪 的类脑相似度
3. 论文分别描述三者之间的联系

4. 实验细节描述
5. 对比实验：PCA组件的个数、三个跟踪之间两两之间的相似度

* Done
0. 选取1500帧图片、fMRI
1. 根据先验脑区（回归出的脑区）提取ROI，再做差分
2. PCA降维，计算相似度

用计算性的方法理解灵长类视觉跟踪在大脑内的形成机制。

* 阶段一：实现跟踪各个脑区和网络的一一对应，并进行消去实验，验证和大脑中某个部位的损伤和删除对应神经网络模块的效果是一致的。

* 阶段二：实现跟踪时神经神经网络的激活响应和灵长类动物大脑中神经元的激活响应类似。

This is an Tensorflow and matlab implementation of single object tracking in videos by using Brain-like Tracking Network, as presented in the following paper:

参考：[A. R. Kosiorek](http://akosiorek.github.io), [A. Bewley](http://ori.ox.ac.uk/mrg_people/alex-bewley/), [I. Posner](http://ori.ox.ac.uk/mrg_people/ingmar-posner/), ["Hierarchical Attentive Recurrent Tracking", NIPS 2017](https://arxiv.org/abs/1706.09262).

* **Author**: Adam Kosiorek, Oxford Robotics Institue, University of Oxford
* **Email**: adamk(at)robots.ox.ac.uk
* **Paper**: https://arxiv.org/abs/1706.09262
* **Webpage**: http://ori.ox.ac.uk/

## Explanation
* checkpoints: model temporary directory when train the BTN.
* gaze: smooth pursuit in forrest gump.
* hart: network structure definition for "Hierarchical Attentive Recurrent Tracking".
* imgs: test img data.
* neurocity: basuc neural network structure.
* predictivePursuit: the code of "A Recurrent Neural Network Based Model of Predictive Smooth Pursuit Eye Movement in Primates".
* result: the output directory
* scripts: some launch scripts
* spem: smooth pursuit eye movements in sinusoidal target stimulus moving horizontally.
* study: some exercise code
* utils: tools code

## Installation

Matlab
Solve "VideoReader" function problem (Could not read file due to an unexpected error. Reason: Unable to initialize the video obtain properties)

https://www.cnblogs.com/walker-lin/p/11520064.html


安装 [Tensorflow v1.14.0] and the following dependencies

    numpy 1.11.3 -> 1.14.6
    /home/d/anaconda2/envs/hart/bin/pip 
    ./pip install tensorflow (1.14.0)
    ./pip install tensorflow==1.2.0 (need libcusolver.so.8.0)
    No module named core_rnn_cell_impl (The code is updated to tensorflow 1.3.0.)
    
    Could not dlopen library 'libcudart.so.10.0';  Cannot dlopen some GPU libraries. Skipping registering GPU devices..
    export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
    source ~/.bashrc 
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    sudo chmod a+r /usr/local/cuda/include/cudnn.h
    sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    cd /usr/local/cuda/lib64
    sudo ln -sf libcudnn.so.7.6.3 libcudnn.so.7 (libcudnn.so.7 is not a symbolic link)
    sudo ln -sf libcudnn.so.7 libcudnn.so
    sudo ldconfig
    sudo ldconfig /usr/local/cuda-10.0/lib64/  (symbolic, force)

 (using `pip install -r requirements.txt` (preferred) or `pip install [package]`):
* matplotlib==1.5.3
* numpy==1.12.1
* pandas==0.18.1
* scipy==0.18.1

    
    predictive pursuit
    Python:
    /data2/whd/workspace/sot/CORnet/venv/bin/python
    /data2/whd/workspace/sot/CORnet/venv/bin/pip
    
    实验室主页：http://dicarlolab.mit.edu/
    
    brain-score
    先安装： result_caching, brainio_collection, brainio_base
    /data2/whd/workspace/sot/CORnet/venv/bin/python ./setup.py install
    
    /data2/whd/workspace/sot/CORnet/venv/bin/pip install xarray==0.12
    botocore

    

## Demo
The notebook `scripts/demo.ipynb` contains a demo, which shows how to evaluate tracker on an arbitrary image sequence. By default, it runs on images located in `imgs` folder and uses a pretrained model.
Before running the demo please download AlexNet weights first (described in the Training section).


## Data

1. [SPM demo data](ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson/wakemandg_hensonrn/)

匿名登录：ftp ftp.mrc-cbu.cam.ac.uk

用户名：anonymous
密码：为空（直接回车）

wget ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson/wakemandg_hensonrn/* --ftp-user=anonymous --ftp-password= -r
    
1. Download KITTI dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). We need [left color images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) and [tracking labels](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip).
2. Unpack data into a data folder; images should be in an image folder and labels should be in a label folder.
3. Resize all the images to `(heigh=187, width=621)` e.g. by using the `scripts/resize_imgs.sh` script.
4. Download [BrainIO](https://github.com/brain-score/brainio_collection/blob/master/brainio_collection/lookup.csv) data

## Training

1. Download the AlexNet weights:
    * Execute `scripts/download_alexnet.sh` or
    * Download the weights from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put the file in the `checkpoints` folder.
2. Run

        python scripts/train_hart_kitti.py --img_dir=path/to/image/folder --label_dir=/path/to/label/folder

The training script will save model checkpoints in the `checkpoints` folder and report train and test scores every couple of epochs. You can run tensorboard in the `checkpoints` folder to visualise training progress. Training should converge in about 400k iterations, which should take about 3 days. It might take a couple of hours between logging messages, so don't worry.

## Evaluation on KITTI dataset
The `scripts/eval_kitti.ipynb` notebook contains the code necessary to prepare (IoU, timesteps) curves for train and validation set of KITTI. Before running the evaluation:
* Download AlexNet weights (described in the Training section).
* Update image and label folder paths in the notebook.

##  Freesurfer
1. Install

        chsh -s /bin/tcsh
        
        d:/data2/whd/software/freesurfer> setenv FREESURFER_HOME /data2/whd/software/freesurfer/
        d:/data2/whd/software/freesurfer> source $FREESURFER_HOME/SetUpFreeSurfer.csh
        
        -------- freesurfer-linux-centos6_x86_64-7.1.1-20200723-8b40551 --------
        Setting up environment for FreeSurfer/FS-FAST (and FSL)
        FREESURFER_HOME   /data2/whd/software/freesurfer/
        FSFAST_HOME       /data2/whd/software/freesurfer//fsfast
        FSF_OUTPUT_FORMAT nii.gz
        SUBJECTS_DIR      /data2/whd/software/freesurfer//subjects
        INFO: /home/d/matlab/startup.m does not exist ... creating
        MNI_DIR           /data2/whd/software/freesurfer//mni

2. Usage

        setenv SUBJECTS_DIR /data2/whd/workspace/sot/hart/result/freesurfer/
        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
        export $LD_LIBRARY_PATH
        
        /data2/whd/software/Qt5.12.9/5.12.9/gcc_64/lib
        
        Cannot load library /data2/whd/software/freesurfer/lib/qt/plugins/platforms/libqxcb.so: (/data2/whd/software/freesurfer/bin/../lib/qt/lib/libQt5Core.so.5: version `Qt_5.12' not found (required by /usr/lib/x86_64-linux-gnu/libQt5XcbQpa.so.5))
QLibraryPrivate::loadPlugin failed on "/data2/whd/software/freesurfer/lib/qt/plugins/platforms/libqxcb.so" : "Cannot load library /data2/whd/software/freesurfer/lib/qt/plugins/platforms/libqxcb.so: (/data2/whd/software/freesurfer/bin/../lib/qt/lib/libQt5Core.so.5: version `Qt_5.12' not found (required by /usr/lib/x86_64-linux-gnu/libQt5XcbQpa.so.5))"

## Polyspace

    polyspace-access -host d -port 9443 -create-project testProject

## Citation

If you find this repo useful in your research, please consider citing:

    @inproceedings{Kosiorek2017hierarchical,
       title = {Hierarchical Attentive Recurrent Tracking},
       author = {Kosiorek, Adam R and Bewley, Alex and Posner, Ingmar},
       booktitle = {Neural Information Processing Systems},
       url = {http://www.robots.ox.ac.uk/~mobile/Papers/2017NIPS_AdamKosiorek.pdf},
       pdf = {http://www.robots.ox.ac.uk/~mobile/Papers/2017NIPS_AdamKosiorek.pdf},
       year = {2017},
       month = {December}
    }


## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see  <http://www.gnu.org/licenses/>.


## Release Notes
**Version 1.0**
* Original version from the paper. It contains the KITTI tracking experiment.

## Error
Step 358468, Data Test loss = 5.9006, loss/obj_iou = 0.2763, weight/att = 3.3323, loss/iou = 0.2433, loss/true = 5.5232, loss/l2 = 0.1032, weight/bbox = 0.3984, weight/l2 = 1.0000, loss/obj_mask = 1.4026, loss/bbox = 2.9466, loss/att = 1.0707, weight/obj_mask = 0.7344, loss/obj_acc = 0.6334, eval time = 15.16s
[[[[-0.00878917 -0.00667416  0.00927724  0.0015092 ]]]]
Step 359161, Data Train loss = nan, loss/obj_iou = 0.0000, weight/att = 3.2936, loss/iou = nan, loss/true = nan, loss/l2 = nan, weight/bbox = nan, weight/l2 = 1.0000, loss/obj_mask = nan, loss/bbox = nan, loss/att = 184.2068, weight/obj_mask = nan, loss/obj_acc = 0.0000, eval time = 115.9s
Step 359161, Data Test loss = nan, loss/obj_iou = 0.0000, weight/att = 3.2936, loss/iou = nan, loss/true = nan, loss/l2 = nan, weight/bbox = nan, weight/l2 = 1.0000, loss/obj_mask = nan, loss/bbox = nan, loss/att = 184.2068, weight/obj_mask = nan, loss/obj_acc = 0.0000, eval time = 14.96s
Traceback (most recent call last):
  File "/data2/whd/workspace/sot/hart/scripts/train_hart_kitti.py", line 272, in <module>
    summary = sess.run(summaries)
  File "/home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 950, in run
    run_metadata_ptr)
  File "/home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1173, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1350, in _do_run
    run_metadata)
  File "/home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1370, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: 2 root error(s) found.
  (0) Invalid argument: Nan in summary histogram for: HierarchicalAttentiveRecurrentTracker/rnn_outputs
	 [[node HierarchicalAttentiveRecurrentTracker/rnn_outputs (defined at data2/whd/workspace/sot/hart/scripts/../hart/model/tracker.py:100) ]]
  (1) Invalid argument: Nan in summary histogram for: HierarchicalAttentiveRecurrentTracker/rnn_outputs
	 [[node HierarchicalAttentiveRecurrentTracker/rnn_outputs (defined at data2/whd/workspace/sot/hart/scripts/../hart/model/tracker.py:100) ]]
	 [[GroupCrossDeviceControlEdges_0/gradients/HierarchicalAttentiveRecurrentTracker/HierarchicalAttentiveRecurrentTracker/while/AttentionCell/pre_DFN/dfn/map/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync/_1758]]
0 successful operations.
0 derived errors ignored.

Errors may have originated from an input operation.
Input Source operations connected to node HierarchicalAttentiveRecurrentTracker/rnn_outputs:
 HierarchicalAttentiveRecurrentTracker/HierarchicalAttentiveRecurrentTracker/TensorArrayStack/TensorArrayGatherV3 (defined at data2/whd/workspace/sot/hart/scripts/../hart/model/tracker.py:94)

Input Source operations connected to node HierarchicalAttentiveRecurrentTracker/rnn_outputs:
 HierarchicalAttentiveRecurrentTracker/HierarchicalAttentiveRecurrentTracker/TensorArrayStack/TensorArrayGatherV3 (defined at data2/whd/workspace/sot/hart/scripts/../hart/model/tracker.py:94)

Original stack trace for u'HierarchicalAttentiveRecurrentTracker/rnn_outputs':
  File "data2/whd/workspace/sot/hart/scripts/train_hart_kitti.py", line 169, in <module>
    is_training=is_training)
  File "data2/whd/workspace/sot/hart/scripts/../hart/model/tracker.py", line 76, in __init__
    super(HierarchicalAttentiveRecurrentTracker, self).__init__(self.__class__.__name__)
  File "data2/whd/workspace/sot/hart/scripts/../neurocity/component/model/model.py", line 56, in __init__
    self._build()
  File "data2/whd/workspace/sot/hart/scripts/../hart/model/tracker.py", line 100, in _build
    tf.summary.histogram('rnn_outputs', outputs)
  File "home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/summary/summary.py", line 179, in histogram
    tag=tag, values=values, name=scope)
  File "home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/ops/gen_logging_ops.py", line 329, in histogram_summary
    "HistogramSummary", tag=tag, values=values, name=name)
  File "home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/framework/op_def_library.py", line 788, in _apply_op_helper
    op_def=op_def)
  File "home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 3616, in create_op
    op_def=op_def)
  File "home/d/anaconda2/envs/hart/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 2005, in __init__
    self._traceback = tf_stack.extract_stack()


Process finished with exit code 1
