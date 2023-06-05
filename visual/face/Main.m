%% Demo code ver. 11/01/2021
%==================================================================================================================================================
% Face Detection in Untrained Deep Neural Networks
% Seungdae Baek, Min Song, Jaeson Jang, Gwangsu Kim & Se-Bum Paik*
%
% *Contact: sbpaik@kaist.ac.kr
%
% Prerequirement 
% 1) MATLAB 2019b or later version is recommended.
% 2) Install deeplearning toolbox. 
% 3) Please download 
% 'Data.zip'(https://zenodo.org/record/5637812/files/Stimulus.zip?download=1), 
% 'Stimulus.zip'(https://zenodo.org/record/5637812/files/Data.zip?download=1) from below link
%
%      - [Data URL] : https://doi.org/10.5281/zenodo.5637812
%
%    and unzip these files in the same directory

% 输出结果和论文图片
% Below results for untrained AlexNet will be shown.
% Result 1) Run_Unit: Spontaneous emergence of face-selectivity in untrained networks (Fig.1, Fig.S1-3)
% Result 2) Run_PFI: Preferred feature images of face-selective units in untrained networks (Fig.2, Fig.S4) 
% Result 3) Run_SVM: Detection of face images using the response of face units in untrained networks (Fig.3, Fig.S11-12) 
% Result 4) Run_Trained: Effect of training on face-selectivity in untrained networks (Fig.4) 
% Result 5) Run_Invariance: Invariant characteristics of face-selective units in untrained networks (Fig.S5) 
% Result 6) Run_View: Viewpoint invariance of face-selective units in untrained networks (Fig.S8) 
%==================================================================================================================================================
close all; clc; clear;
% clc;clear;
seed = 1; rng(seed)                                                       % fixed random seed for regenerating same result

%% 添加数据路径
cur_fullpath = mfilename('fullpath');
path_splits = split(cur_fullpath, 'dong');
data_dir = fullfile(path_splits{1}, 'dong');
proj_dir = data_dir;
data_dir = fullfile(data_dir, 'data');
data_path_splits = split(path_splits{2}, filesep);
for i = 2 : length(data_path_splits)-1
    data_dir = fullfile(data_dir, data_path_splits{i});
end
if ~exist(data_dir, 'dir'); mkdir(data_dir); end

%% 添加路径
if isunix
    model_data_URL = '/data3/data/data/neuro/face';
    stimulus_URL = '/data3/data/data/neuro/face/Stimulus.zip';
else
    model_data_URL = 'E:/data/neuro/face/Data.zip';
    stimulus_URL = 'E:/data/neuro/face/Stimulus.zip';
end
model_data_dir = fullfile(data_dir, 'Data');
if ~exist(model_data_dir, 'dir')
    mkdir(model_data_dir);
    unzip(model_data_URL, data_dir);
end
addpath(model_data_dir);
addpath(fullfile(model_data_dir, 'PretrainedNet'));

stimulus_dir = fullfile(data_dir, 'Stimulus');
if ~exist(stimulus_dir, 'dir')  % 不存在数据则下载并解压到数据目录
    mkdir(stimulus_dir);
    % dataURL = 'https://zenodo.org/record/5637812/files/Stimulus.zip';
    unzip(stimulus_URL, data_dir);
%     gunzip(dataURL, stimulus_dir)                       % creates genres.tar in tempdir
%     untar(fullfile(stimulus_dir, 'genres.tar'), tempdir) % creates genres folder
end
addpath(stimulus_dir);

addpath('Subfunctions');
addpath('Data');
addpath(fullfile(proj_dir, 'utils', 'export_fig-3.25'));

figs_dir = fullfile(fileparts(cur_fullpath), 'latex', 'figs');             % 生成图像的保存目录（用于论文写作）
if ~exist(figs_dir, 'dir'); mkdir(figs_dir); end

toolbox_chk;                                                               % 检查 matlab 版本和工具箱

tic

%% 配置参数
% 示例代码
res1 = 1; res2 = 1; res3 = 1; res4 = 1; res5 = 1; res6 = 1;                % 分析每个对应图的开关
NN = 1;                                                                    % 分析网络的数目
 
% 图像
STR_LABEL = {'Face','Hand','Horn','Flower','Chair','Scrambled'};           % 目标刺激的类别标签
numIMG = 200;                                                              % 目标刺激中 一个类别的图片数目
numCLS = 6;                                                                % 目标刺激中 类别的数目
inpSize = 227;                                                             % 目标刺激中，每张图片的宽度和高度

% 网络
layersSet = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};	               % 特征抽取层的名字
array_sz = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256];         % 每层激活图的维度
layerArray = 1:5;                                                          % 目标层
stdArray = [1 0.01 0.5 1.5 2];                                             % 随机初始化网络时的高斯核标准差
verSet = {'LeCun - Normal dist','LeCun - Uniform dist'};                   % 正态分布；均匀分布 
verArray = [1 2];                                                          % 初始化版本
                                                                           %  1: 正态分布 LeCun / 2: 均匀分布 LeCun uniform 
% 分析                                                                           
pThr = 0.001;                                                              % 选择响应的 p 值门限
idxClass = 1;                                                              % 数据集中面部类别的索引


%% Step 1. 加载预训练 Alexnet 和图片数据集
disp('Load imageset and networks ... (~ 10 sec)');
tic
net = alexnet;                                                             % 预训练的 AlexNet
% analyzeNetwork(net)                                                      % 显示网络架构
% deepNetworkDesigner

% 加载 MAT 文件 filename 中的指定变量
load('IMG_cntr_210521.mat', 'IMG')                                          % 加载目标刺激 (Stimulus/IMG_cntr_210521.mat)
% imshow(IMG(:, :, 1)/255)

toc
% 置换数组维度
IMG = IMG(:,:,1:numIMG*numCLS); IMG_ORI = single(repmat(permute(IMG,[1 2 4 3]),[1 1 3])); clearvars IMG
% IMG = IMG(:,:,:,1:numIMG*numCLS); IMG_ORI = single(IMG); clearvars IMG
% imshow(IMG_ORI(:,:,:,1)/255)

disp('Find face unit in untrained network ... (~ 30 sec)');
Cell_Net = cell(length(verArray), length(stdArray), NN);
Cell_Idx = cell(length(verArray), length(stdArray), NN, length(layerArray));

for nn = 1:NN
    tic
    disp(['%%% Trial : ',num2str(nn),' (',num2str(nn),'/',num2str(NN),')'])
    for vv = 1 %1，不需要改，在 Run_Unit.m 中都遍历了
        disp(['%% Version : ',verSet{vv},' (',num2str(vv),'/',num2str(length(vv)),')'])
        for ss = 1
            disp(['% Weight variation : ',num2str(stdArray(ss)),' (',num2str(ss),'/',num2str(length(ss)),')'])
            %% 步骤 2：加载并产生未训练的 AlexNet
            net_rand = fun_Initializeweight(net, verArray(vv), stdArray(ss));
            
            for ll = length(layerArray)
                %% 步骤 3：在目标层测量神经元的响应
                num_cell = prod(array_sz(layerArray(ll), :));   % 计算每列中元素的乘积
                act_rand = activations(net_rand, IMG_ORI, layersSet{layerArray(ll)});  % 获得 5 个 ReLU 层每一层的激活
                
                %% 步骤 4：对目标类别发现选择性神经元
                [cell_idx] = fun_FindNeuron(act_rand, num_cell, numCLS, numIMG, pThr, idxClass);
                Cell_Idx{vv,ss,nn,ll} = cell_idx; clearvars cell_idx
            end
            Cell_Net{vv,ss,nn} = net_rand; clearvars act_rand net_rand
        end
    end 
    toc
end


if res1 == 0
%% Run_Unit: 在未训练的网络中自发地出现面部选择性
disp('Result 1 ... (~ 10 min)')
tic
Run_Unit;
toc
end


if res2 == 0
%% Run_PFI: 在未训练的网络中，面部选择单元的偏好特征图像（Preferred feature images）
% 有面部选择偏好的单元中的特征图。
disp('Result 2 ... (~ 2 min)')
tic
% 决定仿真类型
% 0 : PFI仿真的快速版本，会显示保存的PFI。
% 1 : 会执行实际的仿真过程，花大概 30 分钟。
Sim = 0;
Run_PFI;
toc
end


if res3 == 0
%% Run_SVM: 在未训练的网络中，使用面部单元的响应进行面部图像的检测
disp('Result 3 ... (~ 5 min)')
tic
Run_SVM;
toc
end


if res4 == 0
%% Run_Trained: 在未训练的网络中，关于面部选择性的训练效果。
disp('Result 4 ... (~ 5 min)')
tic
% 0 : 论文手稿中有关的数据加载（网络数为 10）
% 1 : 训练效果分析的快速版本（网络数为 3） 
SimER = 0;
Run_Trained
toc
end


if res5 == 1  % 需要 15 G内存
%% Run_Invariance: 在未训练网络中，面部选择单元的不变特性
disp('Result 5 ... (~ 5 min)')
tic
% 决定仿真类型 (TT)
% 1 : 不变性分析的快速版本。会显示变换不变性的结果。
% 3 : 会显示变换、大小和旋转不变性结果。这会花大概 10 分钟。
TT = 1;
Run_Invariance; % 请求的 227x227x3x13000 (15.0GB)数组超过预设的最大数组大小(7.9GB)。这可能会导致 MATLAB 无响应。
toc
end


if res6 == 0
%% Run_View: 在未训练网络中，面部选择单元的视角不变性
disp('Result 6 ... (~ 5 min)')
tic
Run_View;
toc
end
