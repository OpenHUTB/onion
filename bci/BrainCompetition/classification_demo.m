%% 脑机接口（Brain Machine Interfaces）
%  实践 - 仅仅包括 SVM 分类，不包括处理猴子的数据


%% 初始化
clear; close all; clc


%% 加载线性数据

fprintf('Loading and Visualizing Data ...\n')

% 从 data1 中加载数据: 工作空间会包含变量 X, y
load('data1.mat');

% 绘制训练数据
plotData(X, y);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% 训练线性 SVM

% Load from data1: 
% You will have X, y in your environment
load('data1.mat');

fprintf('\nTraining Linear SVM ...\n')

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel);
visualizeBoundary(X, y, model);

% fprintf('Program paused. Press enter to continue.\n');
% pause;


%% 可视化数据集 2

fprintf('Loading and Visualizing Data ...\n')

% Load from data2: 
% You will have X, y in your environment
load('data2.mat');

% Plot training data
plotData(X, y);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% 使用 RBF（Radial Basis Function，径向基函数） 核训练 SVM（数据集2）
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% Load from ex6data2: 
% You will have X, y in your environment
load('data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

visualizeBoundary(X, y, model);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% 第三方数据

fprintf('Loading and Visualizing Data ...\n')

% Load from data3: 
% You will have X, y in your environment
load('data3.mat');

% Plot training data
plotData(X, y);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% lots of options...