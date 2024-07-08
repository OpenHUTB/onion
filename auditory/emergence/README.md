
---
# Emergence of music detectors in a deep neural network trained for natural sound recognition

[![bioarxiv](http://img.shields.io/badge/DOI-10.1101/2021.10.27.466049-B31B1B.svg)](https://doi.org/10.1101/2021.10.27.466049)

Authors: Gwangsu Kim<sup>1</sup>, Dong-Kyum Kim<sup>1</sup>, and Hawoong Jeong<sup>1,2</sup><br>

<sup>1</sup> <sub>Department of Physics, KAIST</sub>
<sup>2</sup> <sub>Center for Complex Systems, KAIST</sub>

## 介绍

This repo contains source code for the runs in [Emergence of music detectors in a deep neural network trained for natural sound recognition](https://doi.org/10.1101/2021.10.27.466049)

## Installation

Supported platforms: MacOS and Ubuntu, Python 3.7

Installation using [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html):

```bash
git clone https://github.com/kgspiano/Music.git
cd Music
conda create -y --name music python=3.7
conda activate music
pip install -r requirements.txt
pip install openpyxl
python -m ipykernel install --name music
```

To enable gpu usage, install gpu version `torch` package from [PyTorch](https://pytorch.org).  

## 下载 AudioSet 数据

下载 AudioSet 数据集:
```bash
cd data/AudioSet
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
```
在这些 .csv 文件中，包含每个音频片段的 URL 链接。

## Quickstart

```bash
jupyter notebook
```

Select `music` kernel in the jupyter notebook.


## 参考
[代码链接](https://zenodo.org/records/10081610)
执行
```shell
jupyter nbconvert --to script *.ipynb
```
