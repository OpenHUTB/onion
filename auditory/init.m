% clear; clc;
% 可以改变的配置
%% 路径配置
% 获得工程的路径

raw_url = 'D:\data\neuro\music_genre';  % 本地原始音频和fMRI文件的存放目录
% work_dir = tempdir;
work_dir = 'D:\buffer';  % 工作目录
if ~exist(work_dir, 'dir'); mkdir(work_dir); end

% fMRI原始数据放置的目录
fMRI_dir = fullfile(work_dir, 'data', 'brain', 'auditory', 'music_genre_fMRI', 'preprocess', 'ds003720-download');
if ~exist(fMRI_dir, 'dir'); mkdir(fMRI_dir); end
fMRI_raw_path = fullfile(raw_url, 'ds003720-download.zip');

proj_dir = mfilename("fullpath");
for i = 1 : 3; proj_dir = fileparts(proj_dir); end

% if ~exist(fullfile(proj_dir, 'utils','spr_1_0', ['repadd.' mexext]), 'file')
%     run(fullfile(proj_dir, 'utils', "spr_1_0", "mex_compile.m"));
% end

% https://bicr.atr.jp//cbi/sparse_estimation/sato/VBSR.html
% 稀疏估计工具箱
% addpath(fullfile(proj_dir, 'utils', "spr_1_0"));

% 保存论文所需图片的目录
figs_dir = fullfile(fileparts(mfilename("fullpath")), 'latex', 'figs');


% 大脑解码工具箱
cur_dir = pwd;
brain_decoder_toolbox_dir = fullfile(matlabroot, 'software', 'matlab_utils', 'BrainDecoderToolbox2-0.9.17/');
cd(brain_decoder_toolbox_dir);
run('setpath.m');
cd(cur_dir);


workDir = fullfile(proj_dir, 'data', 'brain', 'auditory') ;
dataDir = fullfile(workDir, 'data');       % 包含大脑和图像特征数据的目录
resultsDir = fullfile(workDir, 'results'); % 保存分析结果的目录
lockDir = fullfile(workDir, 'tmp');        % 保存文件锁（正在分析的过程）的目录

setupdir(dataDir);
setupdir(resultsDir);
setupdir(lockDir);
%% 数据设置

% subjectList  : 受试 IDs 列表（元胞数组）
subjectList  = {'sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005'};
% dataFileList : 在受试列表 `subjectList`中，包含大脑数据的数据文件列表（元胞数组）
dataFileList = {'sub-001.mat', 'sub-002.mat', 'sub-003.mat', 'sub-004.mat', 'sub-005.mat'};
% roiList      : 感兴趣区域列表（元胞数组）
roiList      = {'A1', 'A2', 'A3', 'T23'};
% numVoxelList : 在分析中，对于每个感兴趣区域所包含体素数目的列表（元胞数组）
numVoxelList = { 300,  300,  600,  600};
% featureList  : 图像特征列表（元胞数组），先按照不合并conv3和conv4的测量来测试
featureList  = {'conv1', 'conv2', 'conv3_1', 'conv3_2', ...
                'conv4_1', 'conv4_2'};

% 图像特征数据
imageFeatureFile = 'AudioFeatures.mat';
%% 结果文件名设置

resultFileNameFormat = @(s, r, f) fullfile(resultsDir, sprintf('%s/%s/%s.mat', s, r, f));
%% 加载大脑区域数据

X = load(fullfile(matlabroot, 'software', 'matlab_utils', 'xjview', 'TDdatabase'));
wholeMaskMNIAll = X.wholeMaskMNIAll;

A1_mask = wholeMaskMNIAll.brodmann_area_41;  % 413*3
A2_mask = wholeMaskMNIAll.brodmann_area_42;  % 334*3
A3_mask = wholeMaskMNIAll.brodmann_area_22;  % 1720*3
T2_T3_mask = [wholeMaskMNIAll.brodmann_area_21; wholeMaskMNIAll.brodmann_area_22];  % 3897*3

obj_vox_num = size(A1_mask, 1) + size(A2_mask, 1) + size(A3_mask, 1) + size(T2_T3_mask, 1);  % 6364*3
%% 深度网络

overlapPercentage = 75;