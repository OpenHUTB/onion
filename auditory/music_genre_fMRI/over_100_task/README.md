# 简介
揭示了 大脑中多样性认知函数组织 的量化模型(Quantitative models reveal the organization
of diverse cognitive functions in the brain)。

# 参考文献 
 Nakai & Nishimoto (bioRxiv 2019, doi: https://doi.org/10.1101/614081)

 Nakai & Nishimoto (Nature Communications 2020, doi: https://doi.org/10.1038/s41467-020-14913-w) 

We confirmed that the following MATLAB codes run on MATLAB ver. R2016b, on Ubuntu 14.04.5 LTS. Installation of Freesurfer (v6.0) is required.

# 行为数据的分析
补充图片10
```matlab
BehavAnalysis;
```


# Ridge regression using Task-type and Cognitive-factor models:
 ID = 'sub-01'; % or 'sub-02', …, 'sub-06'
 Ridge_CerebCortex(ID, 'TaskType');
 Ridge_FDRcorr(ID, 'TaskType');

# Determining the best parameter for group PCA/HCA
 Method = 'TaskType';
 Ridge_CerebCortex_RepFit(ID, Method);
 
# Get group ridge weight
 GroupWeight('TaskType');

# Visualization of the hierarchical clustering analysis (Figure 2, S1)
 load('./SampleResult/GroupRidgeWeight.mat','w');
 Cluster_Visualize(w);

# Hierarchical model
 MakeHCAbase
 Ridge_CerebCortex_HCAModel(ID, 'TaskType');
 Ridge_FDRcorr(ID, 'HCA');

# Visualization of the principal component analysis (Figure 3a, b, S2, S3)
 load('./SampleResult/GroupRidgeWeight.mat','w');
 PCAvisualize_RGB(w);
 voxelID = SelectTargVoxelByPCA(ID, TargPC); % Select target voxel based on PCA result (e.g., TargPC = 2)
 PCAvisualize_PosNeg(w, ID, voxelID); % Top voxel of target PC (TargPC, e.g., 1,2,…) is used
 % For the visualization of IPL voxels, use voxel ID = 32850, 32824, and 29651.

# Downloading of reference Neurosynth database is based on the instruction of following link:
 https://github.com/neurosynth/neurosynth
Save the Neurosynth database in the “NSDir”.

# Reverse inference analysis of cognitive factors related to each task cluster (Table 1)
 Cluster = 1; % Target cluster
 Cluster_WeightMap(ID, Cluster);
 NSDir = '~/NeuroSynthDir';
 Cluster_NeuroSynth(ID, Cluster, NSDir);

# Reverse inference analysis of cognitive factors related to each PC (Table S2, S3)
 PC = 1; %Target PC
 PCA_ScoreMap(ID, PC);
 NSDir = '~/NeuroSynthDir';
 PCA_NeuroSynth(ID, PC, NSDir);

# Relative contribution of the top PCs to the largest task clusters (Figure 4c)
 PC2Cluster

# To make cognitive transform function
 NSDir = '~/NeuroSynthDir';
 MakeCTM(ID, NSDir);

# Novel task encoding model analyses (Figure 5, S4, S5)
 NewTask_SaveTargTask;
 Ridge_CerebCortex_NewTask(ID, 'CogFactor');

# Novel task encoding model analyses, with sensorimotor regressors (Figure S6)
 Ridge_CerebCortex_NewTask_WithReg(ID, 'CogFactor');

# Novel task decoding model analysis (Figure 6, S7)
 Ridge_CerebCortex_NewTask_Dec(ID, 'CogFactor');

# Hierarchical clustering analysis using brain responses (Figure S9, S10)
 Cluster_Visualize_raw;
 MDS_Nonlinear;

# Decoding analysis using support vector machine (requirement: LIBSVM v.3.22, https://github.com/cjlin1/libsvm) (Figure S11)
 SVM_raw(ID, 'TaskType');

# Shuffling analysis (Figure S12, S13)
 RidgePart_CerebCortex_NewTask_All_Shuffle(ID,'CogFactor')
 PlotDecoding_Shuffle(ID, 'CogFactor')

# Top voxels analyses
ID = ['sub-0' num2str(ii)]
Ridge_CerebCortex_RepFit(ID, 'CogFactor');
CalcPredAcc_TopVoxels(ID, Method);
Ridge_CerebCortex_NewTask_Dec_TopVoxels(ID, Method);

# For the visualization of PCA scores (PC1-3) and prediction accuracy on the cerebral cortex, we used pycortex (https://github.com/gallantlab/pycortex) on python (ver. 2.7.13). 
Source code of pycortex preprocessing and visualization is given by pcPreProcesss.py and pcVisualize.py, respectively.

# For the extraction of Modulation transfer function (MTF) features, use MakeFeatureSpace_MTF.m included in the “MTFset” folder.
Requirement: gammatonegram (https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/)

# Scripts for the extraction of Motion energy (ME) features are provided elsewhere (https://github.com/gallantlab/motion_energy_matlab).