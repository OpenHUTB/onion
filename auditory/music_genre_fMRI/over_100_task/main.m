% Analysis of behavioral data (Supplementary Figure 10)
BehavAnalysis;

%%
% 当使用最小二乘法计算线性回归模型参数的时候，如果数据集合矩阵（也叫做设计矩阵(design matrix)）X，
% 存在多重共线性，那么最小二乘法对输入变量中的噪声非常的敏感，其解会极为不稳定。
% 为了解决这个问题，就有了脊回归（Ridge Regression ）
% Ridge regression using Task-type and Cognitive-factor models:
ID = 'sub-01'; % or 'sub-02', …, 'sub-06'
addpath('../../../utils/freesurfer');  % Ridge_CerebCortex需要'MRIread'函数
Ridge_CerebCortex(ID, 'TaskType');
Ridge_FDRcorr(ID, 'TaskType');

% Determining the best parameter for group PCA/HCA
Method = 'TaskType';
Ridge_CerebCortex_RepFit(ID, Method);
 
% Get group ridge weight
GroupWeight('TaskType');

% Visualization of the hierarchical clustering analysis (Figure 2, 已经它上面的树状图)
load('./SampleResult/GroupRidgeWeight.mat','w');
Cluster_Visualize(w);

% 层次化模型； Hierarchical model
MakeHCAbase
Ridge_CerebCortex_HCAModel(ID, 'TaskType');
Ridge_FDRcorr(ID, 'HCA');

% PCA可视化； Visualization of the principal component analysis (Figure 3a, b, S2, S3)
load('./SampleResult/GroupRidgeWeight.mat','w');
PCAvisualize_RGB(w);  % Figure 3a
TargPC = 2;
voxelID = SelectTargVoxelByPCA(ID, TargPC); % Select target voxel based on PCA result (e.g., TargPC = 2)
PCAvisualize_PosNeg(w, ID, voxelID); % Top voxel of target PC (TargPC, e.g., 1,2,…) is used
% For the visualization of IPL voxels, use voxel ID = 32850, 32824, and 29651.

% Downloading of reference Neurosynth database is based on the instruction of following link:
% https://github.com/neurosynth/neurosynth
% Save the Neurosynth database in the “NSDir”.
% 下载数据链接：https://github.com/neurosynth/neurosynth-data/archive/refs/tags/0.5.tar.gz
% 怎么解析出需要的.nii.gz文件？

% Reverse inference analysis of cognitive factors related to each task cluster (Table 1)
Cluster = 1; % Target cluster
Cluster_WeightMap(ID, Cluster);
NSDir = '/data3/whd/workspace/data/neurosynth-data-0.5'; 
addpath('../../../utils/toolbox_matlab_nifti');  %  Cluster_NeuroSynth需要MRIvol2vol函数
Cluster_NeuroSynth(ID, Cluster, NSDir);  % 缺少数据“ability_pFgA_pF=0.50_FDR_0.05.nii.gz”

% Reverse inference analysis of cognitive factors related to each PC (Table S2, S3)
PC = 1; %Target PC
PCA_ScoreMap(ID, PC);
NSDir = '~/NeuroSynthDir';
PCA_NeuroSynth(ID, PC, NSDir);

% Relative contribution of the top PCs to the largest task clusters (Figure 4c)
PC2Cluster

% To make cognitive transform function
NSDir = '~/NeuroSynthDir';
MakeCTM(ID, NSDir);

% Novel task encoding model analyses (Figure 5, S4, S5)
NewTask_SaveTargTask;
Ridge_CerebCortex_NewTask(ID, 'CogFactor');

% Novel task encoding model analyses, with sensorimotor regressors (Figure S6)
Ridge_CerebCortex_NewTask_WithReg(ID, 'CogFactor');

% Novel task decoding model analysis (Figure 6, S7)
Ridge_CerebCortex_NewTask_Dec(ID, 'CogFactor');

% Hierarchical clustering analysis using brain responses (Figure S9, S10)
Cluster_Visualize_raw;
MDS_Nonlinear;

% Decoding analysis using support vector machine (requirement: LIBSVM v.3.22, https://github.com/cjlin1/libsvm) (Figure S11)
SVM_raw(ID, 'TaskType');

% Shuffling analysis (Figure S12, S13)
RidgePart_CerebCortex_NewTask_All_Shuffle(ID,'CogFactor')
PlotDecoding_Shuffle(ID, 'CogFactor')

% Top voxels analyses
ID = ['sub-0' num2str(ii)]
Ridge_CerebCortex_RepFit(ID, 'CogFactor');
CalcPredAcc_TopVoxels(ID, Method);
Ridge_CerebCortex_NewTask_Dec_TopVoxels(ID, Method);

% For the visualization of PCA scores (PC1-3) and prediction accuracy on the cerebral cortex, we used pycortex (https://github.com/gallantlab/pycortex) on python (ver. 2.7.13). 
% Source code of pycortex preprocessing and visualization is given by pcPreProcesss.py and pcVisualize.py, respectively.

% For the extraction of Modulation transfer function (MTF) features, use MakeFeatureSpace_MTF.m included in the “MTFset” folder.
% Requirement: gammatonegram (https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/)

% Scripts for the extraction of Motion energy (ME) features are provided elsewhere (https://github.com/gallantlab/motion_energy_matlab).