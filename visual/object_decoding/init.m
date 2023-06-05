
clear;

%% 配置信息
% 运行之前需要修改的路径
utils_dir = fullfile(matlabroot, 'software', 'matlab_utils'); % spr_1_0 和 BrainDecoderToolbox2-0.9.17 所在的目录
workDir = 'D:\data\neuro\object_decoding';  % 原始数据、中间文件、结果文件 所在的目录

%% 目录设置
if ~exist(workDir, 'dir')
    workDir = pwd;
end
dataDir = workDir;       % 包含大脑和图像特征数据的目录
resultsDir = fullfile(workDir, 'results'); % 保存分析结果的目录
lockDir = fullfile(workDir, 'tmp');        % 保存文件锁（正在分析的过程）的目录


%% Path路径
addpath(fullfile(matlabroot, 'software', 'matlab_utils', 'spm12'));

addpath(fileparts(mfilename('fullpath')), 'lib');

% 大脑解码工具箱
cur_dir = pwd;
brain_decoder_toolbox_dir = fullfile(utils_dir, 'BrainDecoderToolbox2-0.9.17/');
cd(brain_decoder_toolbox_dir);
run('setpath.m');
cd(cur_dir);

% https://bicr.atr.jp//cbi/sparse_estimation/sato/VBSR.html
% 稀疏估计工具箱
% 需要使用max_compile进行编译，然后运行： mex -v weight_out_delay_time.c '-compatibleArrayDims'
addpath(fullfile(utils_dir, "spr_1_0"));
% addpath(fullfile('lib', 'SPR_2009_12_17'));
% addpath(fullfile(utils_dir, 'SPR_2011_1111'));  % 报错：函数或变量 'Tall' 无法识别。
