# analysis.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import math
import itertools

import emoji
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics



class Analysis(object):
  
    def __init__(self, res_path='/content/output/test_results.json', n_label=3):
        self.n_label = n_label
        self.res = json.load(open(res_path, 'r'))
        self.B = emoji.emojize(':black_medium_square:')
        self.T = emoji.emojize(':heavy_check_mark:')
        self.F = emoji.emojize(':heavy_multiplication_x:')
    
    def title(self, uid, label, pred, other=''):
        mark = self.T if pred == label else self.F
        print('{0}  ID: {1}\tLabel: {2}\t Pred: {3}\t{4}{5}'.format(self.B, uid, label, pred, mark, other))
    
    def result_list(self):
        for uid, item in self.res.items():
            self.title(uid, item['label'], item['pred'])
      
    def confusion_matrix(self):
        label_list = []
        pred_list = []
        for item in self.res.values():
            label_list.append(item['label'])
            pred_list.append(item['pred'])
        metric_report(label_list, pred_list, self.n_label)
  
    def dist_lines(self):
        for uid, item in self.res.items():
            self.title(uid, item['label'], item['pred'], '\n')
            dist = np.asarray(item['dist'], dtype=np.float32)
            start = np.argmin(np.all(np.absolute(dist-0.5)<1e-3, axis=(1,2,3)))
            end = len(dist)
            length = end-start
            gap = int(0.1*length)
            _, hs, ss, os = dist.shape
            _, axes = plt.subplots(hs, ss, figsize=[ss*9, hs*3.5])
            for h in range(hs):
                for s in range(ss):
                    switch = 0
                    count = 0
                    for f in range(start, end):
                        if dist[f,h,s,s] >= dist[f,h,s,1-s]:
                            count += 1
                        if f > start and ((dist[f,h,s,s] >= 0.5) != (dist[f-1,h,s,s] >= 0.5)):
                            switch += 1
                    per = count / length
                    cat = category(per)
                    axes[h, s].plot(dist[:,h,s,:])
                    axes[h, s].set_xlim(start-gap, end+gap)
                    axes[h, s].set_ylim(-0.1, 1.1)
                    axes[h, s].tick_params(labelsize=17)
                    axes[h, s].grid()
                    axes[h, s].set_title('Head {0}, Modality {1}\n SelfAttenRate {2:.4f}, Category {3}, Switch# {4}'.format(h, s, per, cat, switch), fontsize=17)
                    axes[h, s].legend(['M{}'.format(i) for i in range(os)], fontsize=17)
            plt.tight_layout()
            plt.show()
            print('\n')
      
    def dist_stack_lines(self):
        for uid, item in self.res.items():
            self.title(uid, item['label'], item['pred'], '\n')
            dist = np.asarray(item['dist'], dtype=np.float32)
            sl, hs, ss, os = dist.shape
            _, axes = plt.subplots(hs, ss, figsize=[ss*9, hs*3.5])
            x = list(range(sl))
            for h in range(hs):
                for s in range(ss):
                    y = [dist[:,h,s,i] for i in range(os)]
                    l = ['M{}'.format(i) for i in range(os)]
                    axes[h, s].stackplot(x, *y, labels=l)
                    axes[h, s].set_ylim(0, 1)
                    axes[h, s].set_title('Head {0}, Modality {1}'.format(h, s))
                    axes[h, s].legend(['M{}'.format(i) for i in range(os)])
            plt.tight_layout()
            plt.show()
            print('\n')



def category(percent):
    if percent >= 1.0:
        return 'FIA'
    if percent <= 0.0:
        return 'FOA'
    if percent < 1.0 and percent >= 0.6:
        return 'FOS'
    if percent <= 0.4 and percent > 0.0:
        return 'FIS'
    if percent < 0.6 and percent > 0.4:
        return 'FIOB'



def analysis(res_dir, n_label=3):
    output = {}
    is_json = lambda x: x.endswith('.json')
    for file in filter(is_json, os.listdir(res_dir)):
        path = os.path.join(res_dir, file)
        res = json.load(open(path, 'r'))
        for uid, item in res.items():
            exeid = uid.split('_')[-1]
            if exeid not in output.keys():
                output[exeid] = {}
            dist = np.asarray(item['dist'], dtype=np.float32)
            start = np.argmin(np.all(np.absolute(dist-0.5)<1e-3, axis=(1,2,3)))
            end = len(dist)
            length = end-start
            _, hs, ss, _ = dist.shape
            for h in range(hs):
                for s in range(ss):
                    switch = 0
                    count = 0
                    for f in range(start, end):
                        if dist[f,h,s,s] >= dist[f,h,s,1-s]:
                            count += 1
                        if f > start and ((dist[f,h,s,s] >= 0.5) != (dist[f-1,h,s,s] >= 0.5)):
                            switch += 1
                    per = count / length
                    cat = category(per)
                    if cat not in output[exeid].keys():
                        output[exeid][cat] = {}
                    modal = 'MC' if s == 0 else 'EMG'
                    if modal not in output[exeid][cat].keys():
                        output[exeid][cat][modal] = 0
                    output[exeid][cat][modal] += 1
                    if 'switch' not in output[exeid].keys():
                        output[exeid]['switch'] = {}
                    if modal not in output[exeid]['switch'].keys():
                        output[exeid]['switch'][modal] = []
                    output[exeid]['switch'][modal].append(switch)
    return output



def metric_report(label_list, pred_list, n_label):
    cm = metrics.confusion_matrix(label_list, pred_list, labels=np.arange(n_label))
    accuracy = np.trace(cm) / np.sum(cm)
    mcc = metrics.matthews_corrcoef(label_list, pred_list)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    precision_str = ('[' + ', '.join(['%.4f']*len(precision)) + ']') % tuple(precision)
    precision_avg = np.mean(precision)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    recall_str = ('[' + ', '.join(['%.4f']*len(recall)) + ']') % tuple(recall)
    recall_avg = np.mean(recall)
    f1 = 2*precision*recall/(precision+recall)
    f1_str = ('[' + ', '.join(['%.4f']*len(f1)) + ']') % tuple(f1)
    f1_avg = 2*precision_avg*recall_avg/(precision_avg+recall_avg)
    print('Accuracy:  {}'.format(accuracy))
    print('MCC:       {}'.format(mcc))
    print('Precision: {0},\taverage: {1:.4f}'.format(precision_str, precision_avg))
    print('Recall:    {0},\taverage: {1:.4f}'.format(recall_str, recall_avg))
    print('F1 score:  {0},\taverage: {1:.4f}'.format(f1_str, f1_avg))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    label_names = np.arange(n_label)
    tick_marks = np.arange(n_label)
    plt.xticks(tick_marks, label_names, fontsize=15)
    plt.yticks(tick_marks, label_names, fontsize=15)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:,}'.format(cm[i, j]),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black',
                 fontsize=15)
    plt.tight_layout()
    plt.ylabel('Label', fontsize=15)
    plt.xlabel('Prediction', fontsize=15)
    plt.show()



def cv_result(res_dir=None, n_label=3):
    label_list = []
    pred_list = []
    for f in os.listdir(res_dir):
        if f.endswith('.json'):
            path = os.path.join(res_dir, f)
            res = json.load(open(path, 'r'))
            for item in res.values():
                label_list.append(item['label'])
                pred_list.append(item['pred'])
    metric_report(label_list, pred_list, n_label)



def wilcoxon(x, y=None):

    if y is None:
        d = np.asarray(x)
    else:
        x, y = map(np.asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x - y

    d = np.compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        print("Warning: sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    T = min(r_plus, r_minus)
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    _, repnum = stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = math.sqrt(se / 24)
    Z = (T - mn) / se
    R = abs(Z) / math.sqrt(len(x))
    p = 2. * stats.norm.sf(abs(Z))

    return T, Z, R, p



def wtest_losocv(res1_dir=None, res2_dir=None, metric=metrics.accuracy_score):
    is_json = lambda x: x.endswith('.json')
    files1 = list(filter(is_json, os.listdir(res1_dir)))
    files2 = list(filter(is_json, os.listdir(res2_dir)))
    assert len(files1) == len(files2)

    scores1, scores2 = [], []
    for f1, f2 in zip(sorted(files1), sorted(files2)):
        path1 = os.path.join(res1_dir, f1)
        path2 = os.path.join(res2_dir, f2)
        res1 = json.load(open(path1, 'r'))
        res2 = json.load(open(path2, 'r'))
        lab1 = [it['label'] for it in res1.values()]
        lab2 = [it['label'] for it in res2.values()]
        pred1 = [it['pred'] for it in res1.values()]
        pred2 = [it['pred'] for it in res2.values()]
        scores1.append(metric(lab1, pred1))
        scores2.append(metric(lab2, pred2))

    w, z, r, p = wilcoxon(scores1, scores2)
    print('Wilcoxon signed-rank test, statistic={0}, z={1}, r={2} p-value={3}'.format(w, z, r, p))



def wtest_52cv(res1_dir=None, res2_dir=None, metric=metrics.accuracy_score):
    is_json = lambda x: x.endswith('.json')
    files1 = list(filter(is_json, os.listdir(res1_dir)))
    files2 = list(filter(is_json, os.listdir(res2_dir)))
    assert len(files1) == len(files2)

    diff = np.zeros((5, 2))
    for f1, f2 in zip(sorted(files1), sorted(files2)):
        path1 = os.path.join(res1_dir, f1)
        path2 = os.path.join(res2_dir, f2)
        res1 = json.load(open(path1, 'r'))
        res2 = json.load(open(path2, 'r'))
        lab1 = [it['label'] for it in res1.values()]
        lab2 = [it['label'] for it in res2.values()]
        pred1 = [it['pred'] for it in res1.values()]
        pred2 = [it['pred'] for it in res2.values()]
        score1 = metric(lab1, pred1)
        score2 = metric(lab2, pred2)
        name1 = f1.replace('.json', '')
        name2 = f2.replace('.json', '')
        _, nt1, nf1 = name1.split('_')
        _, nt2, nf2 = name2.split('_')
        assert nt1 == nt2
        assert nf1 == nf2
        diff[int(nt1), int(nf1)] = score1 - score2

    diff = diff.flatten()
    w, z, r, p = wilcoxon(diff)
    print('Wilcoxon signed-rank test, statistic={0}, z={1}, r={2} p-value={3}'.format(w, z, r, p))
