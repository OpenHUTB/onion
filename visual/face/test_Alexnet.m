tic

addpath(fullfile('Data', 'alexnet'));                                     % alexnet模型的路径
addpath('Subfunctions');                                                  % 工具函数的路径


%% 常量设置
seed = 1; rng(seed)                                                       % fixed random seed for regenerating same result

numIMG = 200;                                                              % 每类图片的数目
numCLS = 6;                                                                % 类别数
NN = 1;                                                                    % 所分析的网络数
pThr = 0.001;                                                              % p-value threshold of selective response
idxClass = 1;                                                              % 数据集中人脸类别的索引为1

layerArray = [1:5];                                                        % target layers
layersSet = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};	               % 特征抽取层的名字
array_sz = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256];         % 每一层激活图的维度

verArray = [1 2];                                                          % version of initialization 
vv = 1;                                                                    % 参数初始化的版本(1表示正态分布,2表示均匀分布)
stdArray = [1 0.01 0.5 1.5 2];                                             % std of gaussian kernel for randomly initialized network
ss = 1;                                                                    % 随机初始化高斯核的标准差


%% 加载模型和数据
% 进行norm，使得每一层的输入数据分布一致（即均值为0，方差为1）
if ~exist('net', 'var')
    net = alexnet;
end
net_rand = fun_Initializeweight(net, verArray(vv), stdArray(ss));

if ~exist('IMG_ORI', 'var')
    IMG_ORI = load('Data/IMG_ORI.mat');  IMG_ORI = IMG_ORI.IMG_ORI;  % 227*227*3*(6*200)
end


%% 获得激活
nn = 1;
ll = length(layerArray);  % ll = 5  % 只进行最后一层(第五层)的研究

% 在指定层测量神经元的响应(奇迹发生): 227*227*3*1200 -> 13*13*256*1200
act_rand = activations(net_rand, IMG_ORI, layersSet{layerArray(ll)});             % 获得 5 个 ReLU 层每一层的激活(默认使用GPU加速)

% 找到目标类别(人脸)的选择性单元(奇迹定位)
num_cell = prod(array_sz(layerArray(ll), :));                                     % prod: 计算每列中元素的乘积; 第5层激活图的大小为13*13*256
[cell_idx] = fun_FindNeuron(act_rand, num_cell, numCLS, numIMG, pThr, idxClass);  % 寻找第一个类(人脸)有偏好的神经元
Cell_Idx{vv, ss, nn, ll} = cell_idx; clearvars cell_idx                           % 1*1*1*5 (468*1), 均匀分布初始化参数为290*1


%% 测试激活
num_cell = prod(array_sz(layerArray(5),:));  % 细胞数有43264个
Idx_Face = Cell_Idx{1,1,1,5};                % 人脸的位置id(468*1) 

[rep_mat, rep_mat_3D] = fun_ResZscore(act_rand, num_cell, Idx_Face, numCLS, numIMG);  %　正则化激活到可比较的大小z值(奇迹量化)

fsi_mat = fun_FSI(rep_mat_3D);  % 将每个图片对应的468个位置的向量, 计算其人脸选择性指数: 468*1 <- 468*6*200
mean_fsi = mean(fsi_mat)        % 人脸选择性指数的平均值 0.4068

toc

