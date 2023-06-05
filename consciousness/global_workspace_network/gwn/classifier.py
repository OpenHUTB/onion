# classifier.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import gc
import os
import re
import json
import random
import collections

import emoji
import numpy as np
import tensorflow as tf

import gwn.model as model



class Flag(object):
    """
    Running flags.

    Properties:
    + data_path: str, path of data file
    + ckpt_path: str, path of checkpoint file
    + output_dir: str, directory of output files
    + model_name: str, name of model
    + do_train: bool, whether to do train or not
    + do_valid: bool, whether to do validation or not
    + do_test: bool, whether to do test or not
    + train_ids: list, indexes of train examples
    + valid_ids: list, indexes of validation examples
    + test_ids: list, indexes of test examples
    + padding: bool, whether to pad sequence or not
    + augment: bool, whether to augment data or not
    + balance: bool, whether to do over-sampling or not
    + learning_rate: float, learning rate
    + sample_step: int, step of sampling
    + n_train_epoch: int, number of epochs to train
    + n_warmup_step: int, number of warm up steps
    + train_batch_size: int, batch size for training
    + valid_batch_size: int, batch size for validation
    + test_batch_size: int, batch size for testing
    + save_summary_steps: int, number of steps to save summary
    + save_checkpoints_steps: int, number of steps to save checkpoint
    + keep_checkpoint_max: int, number of checkpoint files to keep
    + log_step_count_steps: int, number of steps to record log messages
    """

    def __init__(self):
        self.data_path = None
        self.ckpt_path = None
        self.output_dir = 'output/'
        self.model_name = 'gwn'
        self.do_train = True
        self.do_valid = True
        self.do_test = True
        self.train_ids = None
        self.valid_ids = None
        self.test_ids = None
        self.padding = True
        self.augment = True
        self.balance = True
        self.learning_rate = 1e-3
        self.sample_step = 1
        self.n_train_epoch = 30
        self.n_warmup_step = 5
        self.train_batch_size = 64
        self.valid_batch_size = 16
        self.test_batch_size = 16
        self.save_summary_steps = 5
        self.save_checkpoints_steps = 5
        self.keep_checkpoint_max = None
        self.log_step_count_steps = 1



class EmoPainExample(object):
    """
    Data structure of a single EmoPain data example.

    Arguments:
    + uid: str, the unique id of this example
    + pose: ndarray, data of pose modality
    + emg: ndarry, data of emg modality
    + label: int, label of this example
    """

    def __init__(self, uid, pose, emg, label):
        self.uid = uid
        self.pose = pose
        self.emg = emg
        self.label = label

    def copy(self, pose_deep=False, emg_deep=False):
        """
        Return a copy of this example.

        Arguments:
        + pose_deep: whether perform deep copy on pose or not
        + emg_deep: whether perform deep copy on emg or not

        Return:
        + object, a new `EmoPainExample` instance
        """

        return EmoPainExample(
            uid=self.uid,
            pose=self.pose.copy() if pose_deep else self.pose,
            emg=self.emg.copy() if emg_deep else self.emg,
            label=self.label
        )



class EmoPainProcessor(object):
    """
    Pre-processing EmoPain data examples.

    Arguments:
    + data_dir: str, directory of json file
    + sample_step: int, the step of sampling on sequence
    """

    def __init__(self, data_dir, sample_step=1):
        data = json.load(open(data_dir, 'r'))
        self.example_dict = {}
        self.label_dict = {}
        self.max_length = 0
        pid_set = set()
        for pdid, motion_dict in data.items():
            examples = []
            for mid, content in motion_dict.items():
                uid = '{0}_{1}'.format(pdid, mid)
                pose = content['pose'][::sample_step]
                emg = content['emg'][::sample_step]
                label = self.label_class(content['pain'])
                examples.append(EmoPainExample(
                    uid=uid,
                    pose=np.asarray(pose, dtype=np.float32),
                    emg=np.asarray(emg, dtype=np.float32),
                    label=label
                ))
                if label in self.label_dict.keys():
                    self.label_dict[label].append(uid)
                else:
                    self.label_dict[label] = [uid]
                pose_len, emg_len = len(pose), len(emg)
                assert pose_len == emg_len
                if pose_len >= self.max_length:
                    self.max_length = pose_len
            if len(examples) == 0:
                continue
            pid = pdid.split('_')[0]
            pid_set.add(pid)
            if pid in self.example_dict.keys():
                self.example_dict[pid].extend(examples)
            else:
                self.example_dict[pid] = examples
        self.pid_list = list(pid_set)
        self.pid_list.sort()

    @staticmethod
    def label_class(num):
        """
        Convert label values to classes.

        Arguments:
        + num: float, the true label value

        Return:
        + int, index of class
        """
        if num == 0.0:
            return 0
        else:
            return 1

    @staticmethod
    def get_n_label():
        """
        Get number of labels.

        Return:
        + int, number of labels
        """
        return 2

    @staticmethod
    def rotate(seq, rad):
        """
        Rotate pose data in 3D space.

        Arguments:
        + seq: ndarray, the input sequence
        + rad: float, angle to rotate in radians

        Return:
        + ndarray, sequence after rotation
        """
        rm = np.array([
            [np.cos(rad),  0., np.sin(rad)],
            [0.,           1.,          0.],
            [-np.sin(rad), 0., np.cos(rad)]
        ], dtype=np.float32) # along y-axis
        res = seq.reshape(len(seq), 3, -1)
        res = np.matmul(rm, res)
        res = res.reshape(len(res), -1)
        return res

    @staticmethod
    def write_examples(examples, output_path):
        """
        Write examples to disk.

        Arguments:
        + examples: list, example instances
        + output_path: str, the path of saved file
        """
        writer = tf.io.TFRecordWriter(output_path)
        for example in examples:
            features = collections.OrderedDict()
            features['pose'] = tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=feat))
                for feat in example.pose
            ])
            features['emg'] = tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=feat))
                for feat in example.emg
            ])
            label = tf.train.Feature(int64_list=tf.train.Int64List(value=[example.label]))
            tf_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={'label': label}),
                feature_lists=tf.train.FeatureLists(feature_list=features)
            )
            writer.write(tf_example.SerializeToString())
        writer.close()

    def _pad(self, examples):
        """ Pad the examples. """
        for example in examples:
            pose_shape = example.pose.shape
            emg_shape = example.emg.shape
            assert pose_shape[0] == emg_shape[0]
            diff = self.max_length - pose_shape[0]
            pose_patch = np.zeros((diff, pose_shape[1]))
            emg_patch = np.zeros((diff, emg_shape[1]))
            example.pose = np.concatenate([pose_patch, example.pose])
            example.emg = np.concatenate([emg_patch, example.emg])
            yield example

    def _aug(self, examples):
        """ Augment the examples """
        for example in examples:
            raw = example.copy()
            raw.uid += '_o'
            yield raw
            ro_90 = example.copy(1, 0)
            ro_90.uid += '_90'
            ro_90.pose = self.rotate(ro_90.pose, np.pi/2.)
            yield ro_90
            ro_180 = example.copy(1, 0)
            ro_180.uid += '_180'
            ro_180.pose = self.rotate(ro_180.pose, np.pi)
            yield ro_180
            ro_270 = example.copy(1, 0)
            ro_270.uid += '_270'
            ro_270.pose = self.rotate(ro_270.pose, -np.pi/2.)
            yield ro_270

    def _bal(self, examples):
        """ Balance the examples """
        for example in examples:
            for _ in range(self.bal_dict[example.uid]):
                yield example

    def get_pose_feat_size(self):
        return self.example_dict[self.pid_list[0]][0].pose.shape[-1]

    def get_emg_feat_size(self):
        return self.example_dict[self.pid_list[0]][0].emg.shape[-1]

    def get_id_list(self):
        return self.pid_list

    def get_max_seq_length(self):
        return self.max_length

    def get_examples(self, to_file=None, ids=None, pad=False, aug=False, bal=False, shuffle=False):
        """
        Get a list of examples.

        Arguments:
        + to_file: str or None, the path to save examples, None for return directly
        + ids: list, indexes of examples to load
        + pad: bool, whether to pad examples or not
        + aug: bool, whether to augment examples or not
        + bal: bool, whether to over-sampling on minor examples or not
        + shuffle: bool, whether to shuffle the examples or not

        Return:
        + list or int, if to_file=None, return a list of examples,
            otherwise return the number of examples
        """
        if ids is None:
            ids = list(range(len(self.pid_list)))
        for i in ids:
            assert i in range(len(self.pid_list))
        if bal:
            max_label = max([len(v) for v in self.label_dict.values()])
            self.bal_dict = {}
            for uids in self.label_dict.values():
                n_rep = max_label // len(uids)
                n_res = max_label % len(uids)
                for uid in uids:
                    self.bal_dict[uid] = n_rep
                for uid in random.sample(uids, k=n_res):
                    self.bal_dict[uid] += 1
        examples = []
        for i in ids:
            pid = self.pid_list[i]
            res = self.example_dict[pid]
            if bal:
                res = self._bal(res)
            if aug:
                res = self._aug(res)
            if pad:
                res = self._pad(res)
            examples.extend(res)
        if shuffle:
            random.shuffle(examples)
        gc.collect()
        if to_file is not None:
            self.write_examples(examples, to_file)
            return len(examples)
        else:
            return examples



def input_fn_builder(examples, seq_length, batch_size, is_training):
    """
    Build input function for estimator.

    Arguments:
    + examples: list, input examples
    + seq_length: int, sequence length of examples
    + batch_size: int, batch size
    + is_training: bool, whether is training or not

    Return:
    + func, input function to be passed to estimator
    """

    pose_feat_size = examples[0].pose.shape[1]
    emg_feat_size = examples[0].emg.shape[1]

    pose_examples = []
    emg_examples = []
    labels = []

    for example in examples:
        pose_examples.append(example.pose)
        emg_examples.append(example.emg)
        labels.append(example.label)

    def input_fn(params):
        n_example = len(examples)
        d = tf.data.Dataset.from_tensor_slices({
            'pose': tf.constant(
                value=np.asarray(pose_examples, dtype=np.float32),
                shape=[n_example, seq_length, pose_feat_size],
                dtype=tf.float32
            ),
            'emg': tf.constant(
                value=np.asarray(emg_examples, dtype=np.float32),
                shape=[n_example, seq_length, emg_feat_size],
                dtype=tf.float32
            ),
            'label': tf.constant(
                value=labels,
                shape=[n_example],
                dtype=tf.int32
            )
        })
        if is_training:
            d = d.repeat()
            d = d.shuffle(10*batch_size)
        d = d.batch(batch_size)
        return d

    return input_fn



def tf_record_input_fn_builder(record_path, pose_feat_size, emg_feat_size, batch_size, is_training):
    """
    Build input function that load tf record file from disk.

    + record_path: str, path of tf record file
    + pose_feat_size: int, feature size of pose data
    + emg_feat_size: int, feature size of emg data
    + batch_size: int, batch size
    + is_training: bool, whether is training or not

    Return:
    + func, input function to be passed to estimator
    """

    context_feat = {
        'label': tf.FixedLenFeature([], tf.int64)
    }
    sequence_feat = {
        'pose': tf.FixedLenSequenceFeature([pose_feat_size], tf.float32),
        'emg': tf.FixedLenSequenceFeature([emg_feat_size], tf.float32)
    }

    def map_func(record):
        res = tf.io.parse_single_sequence_example(
            serialized=record,
            context_features=context_feat,
            sequence_features=sequence_feat
        )
        return {**res[0], **res[1]}

    def input_fn(params):
        d = tf.data.TFRecordDataset(record_path)
        if is_training:
            d = d.repeat()
            d = d.shuffle(10*batch_size)
        d = d.map(map_func)
        d = d.batch(batch_size)
        return d

    return input_fn



def create_model(inputs, labels, config, is_training, n_label, model_name):
    """
    Create downstream task model.

    Arguments:
    + inputs: list, a list of input tensors for different modalities
    + labels: list, corresponding labels of inputs
    + config: object, instance of `GWNConfig`
    + is_training: bool, whether is training or not
    + n_label: int, number of labels
    + model_name: str, name of model

    Return:
    + tensor (), loss of model
    + tensor (B,), loss of each example
    + tensor (B, L), log probability
    + tensor (S, B, N, [1, M], M), attention distributions
    """

    model_names = ['pretrain', 'conc', 'gwn']
    if model_name not in model_names:
        raise ValueError('Unknown model name %s. Accepted models are: %s.'.format(model_name, model_names))
    
    if model_name == 'pretrain':
        md = model.MappingPretrain(
            inputs=inputs,
            config=config,
            is_training=is_training
        )
        return md.get_loss(), None, None, None

    if model_name == 'conc':
        md = model.ConcModel(
            inputs=inputs,
            config=config,
            is_training=is_training
        )
        dists = tf.zeros_like(labels) # (B, L)

    if model_name == 'gwn':
        md = model.GWNModel(
            inputs=inputs,
            config=config,
            is_training=is_training
        )
        dists = md.get_dist_sequence() # (S, B, N, [1, M], M)

    output = md.get_projection() # (B, P)

    with tf.variable_scope('loss'):
        if is_training:
            output = tf.nn.dropout(output, rate=0.1)
        logits = tf.keras.layers.Dense(n_label)(output) # (B, L)
        log_prob = tf.nn.log_softmax(logits, axis=-1) # (B, L)

        one_hot_labels = tf.one_hot(labels, depth=n_label, dtype=tf.float32) # (B, L)

        per_sample_loss = -tf.reduce_sum(one_hot_labels*log_prob, axis=-1) # (B,)
        loss = tf.reduce_mean(per_sample_loss)

    return loss, per_sample_loss, log_prob, dists



def create_optimizer(loss, init_lr, n_train_step, n_warmup_step):
    """
    Create train op for estimator.

    Arguments:
    + loss: tensor (), loss of downstream model
    + init_lr: float, initial learning rate
    + n_train_step: int, number of total training steps
    + n_warmup_step: int, number of warm up steps

    Return:
    + object, train op
    """

    gs = tf.train.get_or_create_global_step()
    lr = tf.constant(init_lr, shape=[], dtype=tf.float32)

    lr = tf.train.polynomial_decay(
        learning_rate=lr,
        global_step=gs,
        decay_steps=n_train_step
    )

    if n_warmup_step > 0:
        gs_int = tf.cast(gs, tf.int32)
        ws_int = tf.constant(n_warmup_step, dtype=tf.int32)
        gs_float = tf.cast(gs_int, tf.float32)
        ws_float = tf.cast(ws_int, tf.float32)
        wpd = gs_float / ws_float
        wlr = init_lr * wpd
        is_w = tf.cast(gs_int < ws_int, tf.float32)
        lr = (1.0 - is_w) * lr + is_w * wlr

    op = tf.train.AdamOptimizer(lr).minimize(loss, gs)

    return op



def model_fn_builder(config, n_label, model_name, lr, n_train_step, n_warmup_step):
    """
    Build model function for estimator.

    Arguments:
    + config: object, instance of `GWNConfig`
    + n_label: int, number of labels
    + model_name: str, name of model
    + lr: float, learning rate
    + n_trainig_step: number of training steps
    + n_warmup_step: number of warmup steps

    Return:
    + func, model function to be passed to estimator
    """

    def model_fn(features, labels, mode, params):
        pose = features['pose']
        emg = features['emg']
        label = features['label']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss, per_sample_loss, log_prob, dists = create_model(
            inputs=[pose, emg],
            labels=label,
            config=config,
            is_training=is_training,
            n_label=n_label,
            model_name=model_name
        )

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info('***************************')
            tf.logging.info('*** Trainable Variables ***')
            tf.logging.info('***************************')
            for var in tf.trainable_variables():
                tf.logging.info('  name = {0}, shape= {1}'.format(var.name, var.shape))
            train_op = create_optimizer(loss, lr, n_train_step, n_warmup_step)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            pred = tf.argmax(log_prob, axis=-1, output_type=tf.int32)
            acc = tf.metrics.accuracy(labels=label, predictions=pred)
            los = tf.metrics.mean(values=per_sample_loss)
            eval_metric_ops = {'eval_acc': acc, 'eval_loss': los}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )
        else:
            pred = tf.argmax(log_prob, axis=-1, output_type=tf.int32)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'pred': pred, 'dists': dists}
            )

        return output_spec

    return model_fn



def main(FLAG):
    """
    Main procedure.

    Arguments:
    + FLAG: object, instance of `Flag`
    """

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAG.output_dir)

    epp = EmoPainProcessor(FLAG.data_path, FLAG.sample_step)

    n_train_step = None
    if FLAG.do_train:
        train_record_path = os.path.join(FLAG.output_dir, 'train.tfrecord')
        n_train_example = epp.get_examples(
            to_file=train_record_path,
            ids=FLAG.train_ids,
            pad=FLAG.padding,
            aug=FLAG.augment,
            bal=FLAG.balance,
            shuffle=True
        )
        n_train_step = int(n_train_example/FLAG.train_batch_size*FLAG.n_train_epoch)
        train_input_fn = tf_record_input_fn_builder(
            record_path=train_record_path,
            pose_feat_size=epp.get_pose_feat_size(),
            emg_feat_size=epp.get_emg_feat_size(),
            batch_size=FLAG.train_batch_size,
            is_training=True
        )

    if FLAG.do_valid:
        valid_examples = epp.get_examples(
            to_file=None,
            ids=FLAG.valid_ids,
            pad=FLAG.padding,
            aug=False,
            bal=False,
            shuffle=False
        )
        valid_input_fn = input_fn_builder(
            examples=valid_examples,
            seq_length=epp.get_max_seq_length(),
            batch_size=FLAG.valid_batch_size,
            is_training=False
        )

    if FLAG.do_test:
        test_examples = epp.get_examples(
            to_file=None,
            ids=FLAG.test_ids,
            pad=FLAG.padding,
            aug=False,
            bal=False,
            shuffle=False
        )
        test_input_fn = input_fn_builder(
            examples=test_examples,
            seq_length=epp.get_max_seq_length(),
            batch_size=FLAG.test_batch_size,
            is_training=False
        )

    model_fn = model_fn_builder(
        config=model.GWNConfig(),
        n_label=epp.get_n_label(),
        model_name=FLAG.model_name,
        lr=FLAG.learning_rate,
        n_train_step=n_train_step,
        n_warmup_step=FLAG.n_warmup_step
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAG.output_dir,
        save_summary_steps=FLAG.save_summary_steps,
        save_checkpoints_steps=FLAG.save_checkpoints_steps,
        keep_checkpoint_max=FLAG.keep_checkpoint_max,
        log_step_count_steps=FLAG.log_step_count_steps
    )

    warm_config = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAG.ckpt_path,
        vars_to_warm_start='{}_model/mapping/*'.format(FLAG.model_name)
    ) if FLAG.ckpt_path and FLAG.do_train else None

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        warm_start_from=warm_config
    )

    if FLAG.do_train and FLAG.do_valid:
        tf.logging.info('*******************************************')
        tf.logging.info('***** Running Training and Validation *****')
        tf.logging.info('*******************************************')
        tf.logging.info('  Train num examples = {}'.format(n_train_example))
        tf.logging.info('  Eval num examples = {}'.format(len(valid_examples)))
        tf.logging.info('  Train batch size = {}'.format(FLAG.train_batch_size))
        tf.logging.info('  Eval batch size = {}'.format(FLAG.valid_batch_size))
        tf.logging.info('  Num steps = {}'.format(n_train_step))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=n_train_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, start_delay_secs=0, throttle_secs=0)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAG.do_train and not FLAG.do_valid:
        tf.logging.info('****************************')
        tf.logging.info('***** Running Training *****')
        tf.logging.info('****************************')
        tf.logging.info('  Num examples = {}'.format(n_train_example))
        tf.logging.info('  Batch size = {}'.format(FLAG.train_batch_size))
        tf.logging.info('  Num steps = {}'.format(n_train_step))
        estimator.train(input_fn=train_input_fn, max_steps=n_train_step)

    if FLAG.do_valid and not FLAG.do_train:
        tf.logging.info('******************************')
        tf.logging.info('***** Running Validation *****')
        tf.logging.info('******************************')
        tf.logging.info('  Num examples = {}'.format(len(valid_examples)))
        tf.logging.info('  Batch size = {}'.format(FLAG.valid_batch_size))
        result = estimator.evaluate(input_fn=valid_input_fn, checkpoint_path=FLAG.ckpt_path)
        output_valid_file = os.path.join(FLAG.output_dir, 'valid_results.txt')
        with tf.gfile.GFile(output_valid_file, 'w') as writer:
            tf.logging.info('***** Validation Results *****')
            for key in sorted(result.keys()):
                tf.logging.info('  {0} = {1}'.format(key, str(result[key])))
                writer.write('{0} = {1}\n'.format(key, str(result[key])))

    if FLAG.do_test:
        tf.logging.info('***************************')
        tf.logging.info('***** Running Testing *****')
        tf.logging.info('***************************')
        tf.logging.info('  Num examples = {}'.format(len(test_examples)))
        tf.logging.info('  Batch size = {}'.format(FLAG.test_batch_size))
        result = estimator.predict(input_fn=test_input_fn, checkpoint_path=FLAG.ckpt_path)
        output_test_file = os.path.join(FLAG.output_dir, 'test_results.json')
        with tf.gfile.GFile(output_test_file, 'w') as writer:
            tf.logging.info('***** Test Results *****')
            output = {}
            for i, pred in enumerate(result):
                uid = test_examples[i].uid
                lab = int(test_examples[i].label)
                res = int(pred['pred'])
                dis = pred['dists'].tolist()
                emo = emoji.emojize(':heavy_check_mark:' if lab == res else ':heavy_multiplication_x:')
                tf.logging.info('  ID: {0}\tLabel: {1}\tPrediction: {2}\t{3}'.format(uid, lab, res, emo))
                output[uid] = {'label': lab, 'pred': res, 'dist': dis}
            writer.write(json.dumps(output))



if __name__ == "__main__":
    main(Flag())
