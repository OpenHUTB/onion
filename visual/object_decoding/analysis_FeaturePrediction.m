% analysis_FeaturePrediction    
% 运行特征预测（根据fMRI来推测特征，并推断图片的类别）


% 运行之前需要删除之前临时目录中的文件锁，不然results\Subject1\V1中会没有cnn1.mat-cnn5.mat
if exist("resultsDir", 'dir'); rmdir(resultsDir, 's'); end
if exist("lockDir", "dir"); rmdir(fullfile(workDir, 'tmp'), 's'); end

run(fullfile(fileparts(mfilename('fullpath')), 'init.m'));

%% 初始化设置

%% 数据设置
% subjectList  : 受试 IDs 列表（元胞数组）
subjectList  = {'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'};
% dataFileList : 在受试列表 `subjectList`中，包含大脑数据的数据文件列表（元胞数组）
dataFileList = {'Subject1.mat', 'Subject2.mat', 'Subject3.mat', 'Subject4.mat', 'Subject5.mat'};
% roiList      : 感兴趣区域列表（元胞数组）
roiList      = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};
% numVoxelList : 在分析中，对于每个感兴趣区域所包含体素数目的列表（元胞数组）
numVoxelList = { 500,  500,  500,  500,   500,   500,   500,  1000,  1000,  1000};
% featureList  : 图像特征列表（元胞数组）
featureList  = {'cnn1', 'cnn2', 'cnn3', 'cnn4', ...
                'cnn5', 'cnn6', 'cnn7', 'cnn8', ...
                'hmax1', 'hmax2', 'hmax3', 'gist', 'sift'};

% 图像特征数据
imageFeatureFile = 'ImageFeatures.mat';


%% 结果文件名设置
resultFileNameFormat = @(s, r, f) fullfile(resultsDir, sprintf('%s/%s/%s.mat', s, r, f));

%% 模型参数
nTrain = 200; % 总训练迭代的次数
nSkip  = 200; % 显示信息跳过的步数

%% 主分析
% 注意：在原始论文中，训练迭代的次数（`nTrain`）为 2000
fprintf('%s started\n', mfilename);


%% 初始化
addpath(genpath('./lib'));

setupdir(resultsDir);
setupdir(lockDir);


%% 加载数据

%% 加载大脑数据（fMRI）
fprintf('Loading brain data...\n');

% 每个人的v1的roi不一样
% 统计V1 体素的个数：
% values = metadata.value; sum(values(9,:), 'omitnan')
% 受试1: 1004
% 受试2: 757
% 受试3: 872
% 受试4: 719
% 受试5: 659
for n = 1:length(subjectList)
    % 3450x4375, {key:1*21, description:1*21, value:21x4375}，3450个fMRI图像（对应3450张图像刺激：1200+1750+500）
    % 21个维度，所有ROI区域有4375个体素（实际只有4370个体素，前2列和后3列不是不是体素值）
    % 第1维（数据类型：1是训练数据、2是感知测试数据、3是想像测试数据）只需要第一个体素标记就行，其他为NaN；
    % 第2维（Run number）为 1
    % 第3维：1 = Label (image ID)，最后3列为1？
    % 第4维（1 = Voxel data）表示如果为体素则该维度标记为1，不是体素标记为NaN（前2列和后3列不是不是体素值）
    % 第5-7维：分别表示体素的 x、y、z 坐标值；
    % 第8维：volumn的索引（做实验时取的脑区快照的id？）
    % 第9维（V1的感兴趣区域）表示全脑4375个体素中是V1的就标记为1（即V1的掩膜）；
    % 第9-18维：'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'
    % FFA：梭状回面孔区、LOCk：外侧枕骨复合体、PPA：海马旁回；
    % 第19-21行：这3行的最后3列为1的对角矩阵？
    [dataset, metadata] = load_data(fullfile(dataDir, dataFileList{n}));

    dat(n).subject = subjectList{n};
    dat(n).dataSet = dataset;
    dat(n).metaData = metadata;
end

%% 加载图片特征（matconvnet提取的特征）
fprintf('Loading image feature data...\n');

% feat.dataSet:  16622x13027
% feat.metaData: {1x17, 1x17, 17x13027}
% feat.metaData.value: 17*13027
% 第1-8维（行）：cnn1-cnn8的特征，不是cnn1的特征用NaN填充
% 第9-11维：hmax1-3
% 第12维：gist特征
% 第13维：sift特征 
% 第14维：Image ID
% 第15维：Category ID
% 第16维：1 = 图片类型 (1 = 训练图片; 2 = 测试图片; 3 = test category averaged; 4 = novel (candidate) category averaged
% 第17维：在每一层中单元的索引
[feat.dataSet, feat.metaData] = load_data(fullfile(dataDir, imageFeatureFile));


%% 创建 分析参数 矩阵 (analysisParam)
analysisParam = uint16(zeros(length(subjectList) * length(roiList) * length(featureList), 3));

c = 1;
for iSbj = 1:length(subjectList)
for iRoi = 1:length(roiList)
for iFeat = 1:length(featureList)
    analysisParam(c, :) = [iSbj, iRoi, iFeat];
    c =  c + 1;
end
end
end

if c < size(analysisParam, 1)
    analysisParam(c:end, :) = [];
end


%% 分析循环（受试数*ROI数*目标网络层数）
for n = 1:size(analysisParam, 1)

    %% 初始化

    % 在当前分析中获得数据索引（650*3）
    iSbj = analysisParam(n, 1);   % 受试的ID
    iRoi = analysisParam(n, 2);   % 脑区ROI的ID
    iFeat = analysisParam(n, 3);  % 目标网络层的ID

    % 设置分析ID 和 结果文件名
    analysisId = sprintf('%s-%s-%s-%s', ...
                         mfilename, ...
                         subjectList{iSbj}, ...
                         roiList{iRoi}, ...
                         featureList{iFeat});
    resultFile = resultFileNameFormat(subjectList{iSbj}, ...
                                      roiList{iRoi}, ...
                                      featureList{iFeat});

    % 检查是否重复运行
    if checkfiles(resultFile)
        % 分析结果已经存在
        fprintf('Analysis %s is already done and skipped\n', analysisId);
        continue;
    end

    if islocked(analysisId, lockDir)
        % 分析正在运行
        fprintf('Analysis %s is already running and skipped\n', analysisId);
        continue;
    end

    fprintf('Start %s\n', analysisId);
    lockcomput(analysisId, lockDir);

    %% 加载数据

    %% 得到大脑数据
    voxSelector = sprintf('ROI_%s = 1', roiList{iRoi});
    nVox = numVoxelList{iRoi};

    brainData = select_data(dat(iSbj).dataSet, dat(iSbj).metaData, voxSelector);

    dataType = get_dataset(dat(iSbj).dataSet, dat(iSbj).metaData, 'DataType');
    labels = get_dataset(dat(iSbj).dataSet, dat(iSbj).metaData, 'Label');
    labels = labels(:,1);

    % dataType
    % --------
    %
    % - 1: 训练数据
    % - 2: 测试数据 (感知)
    % - 3: 测试数据 (想像)
    %

    % 获得训练和测试的大脑数据
    indTrain = dataType == 1;       % 训练数据的索引
    indTestPercept = dataType == 2; % 感知测试数据的索引
    indTestimagery = dataType == 3; % 想像测试数据的索引

    trainData = brainData(indTrain, :);
    testPerceptData = brainData(indTestPercept, :);
    testImageryData = brainData(indTestimagery, :);

    trainLabels = labels(indTrain, :);
    testPerceptLabels = labels(indTestPercept, :);
    testimageryLabels = labels(indTestimagery, :);

    %% 获得图像特征
    layerFeat = select_data(feat.dataSet, feat.metaData, ...
                            sprintf('%s = 1', featureList{iFeat}));
    featType = get_dataset(feat.dataSet, feat.metaData, 'FeatureType');
    imageIds = get_dataset(feat.dataSet, feat.metaData, 'ImageID');

    % featType
    % --------
    %
    % - 1 = 训练
    % - 2 = 测试
    % - 3 = 测试类别
    % - 4 = 其他类别
    %

    % 获得训练和测试的图像特征
    trainFeat = layerFeat(featType == 1, :);  % 16622x1000 -> 1200x1000
    trainImageIds = imageIds(featType == 1, :);  % 16622x1 -> 1200x1 

    trainFeat = get_refdata(trainFeat, trainImageIds, trainLabels); % 1200x1000 <- (1200x1000,1200x1,1200x1)

    %% 预处理
    %% 正则化大脑数据 1200x1004
    [trainData, xMean, xNorm] = zscore(trainData);  % 将输入矩阵 按列 变换到 均值为0，标准差为1

    testPerceptData = bsxfun(@rdivide, bsxfun(@minus, testPerceptData, xMean), xNorm); % 1750x1004
    testImageryData = bsxfun(@rdivide, bsxfun(@minus, testImageryData, xMean), xNorm); % 500x1004

    %% 正则化图像特征
    [trainFeat, yMean, yNorm] = zscore(trainFeat);  % 1200x1000

    %% 特征预测
    predictPercept = [];  % 感知测试的预测标签
    predictImagery = [];  % 想像测试的预测标签

    numUnits = size(trainFeat, 2);  % 当前脑区的体素数：1000
    % numUnits = 100;  % 减少单元数，进行快速测试

    % 训练集：150个类别 * 每个类别8张图像 = 1200张图（每个只呈现一次）
    for i = 1:numUnits  % 对每个体素都构建一个线性模型进行特征预测
        fprintf('Unit %d\n', i);

        %% 获得当前单元的特征
        yTrain = trainFeat(:, i);  % 1200x1 <- 1200x1000

        %% 基于相关性进行体素的选择
        % 在回归分析中，在 训练图片会话中，对目标变化 显示出最高相关系数 的体素 ，被选来预测每个特征。
        % 对V1-V4, LOC, FFA 和 PPA 最多500个体素；对LVC, HVC 和 VC 最多1000个体素
        cor = fastcorr(trainData, yTrain); % 1004x1 <- (1200x1004,1200x1)
        % 根据1200张图像的y值，从1004个体素中选择相关性最高的nVox(500)个体素
        [xTrain, selInd] = select_top(trainData, abs(cor), nVox); % [1200x500,1x1004] <- (1200x1004,1004x1,500)
        % 测试集：50个类别 x 每个类别 1 张图像（每个呈现35次）
        xTestPercept = testPerceptData(:, selInd);  % 1750x500 <- 1750*1004(:, 1*1004)
        % 直观地想象来自 50 个类别中的 1 个类别的图像（20 个运行 x 每次运行 25 个类别 = 500；每个类别 10 个样本）
        xTestImagery = testImageryData(:, selInd);  % 500 x500 <- 500 *1004(:, 1*1004)

        %% 为稀疏线性回归（SLR）函数添加偏置项和转置矩阵
        xTrain = add_bias(xTrain)';              % 501x1200 <- 1200x500
        xTestPercept = add_bias(xTestPercept)';  % 501x1750 <- 1750x500
        xTestImagery = add_bias(xTestImagery)';  % 501x500  <- 500x500

        yTrain = yTrain';  % 1x1200 <- 1200x1

        %% 图像特征解码
        % 模型参数
        param.Ntrain = nTrain;  % 训练的迭代次数
        param.Nskip  = nSkip;   % 跳过打印的数目
        param.data_norm = 1;
        param.num_comp = nVox;

        param.xmean = xMean;
        param.xnorm = xNorm;
        param.ymean = yMean(i);
        param.ynorm = yNorm(i);

        % 模型训练（500个体素 在1200张图像刺激时 xTrain 的响应 yTrain）
        % y = w*x + b（x 表示 V1 有 500 个体素）
        model = linear_map_sparse_cov(xTrain, yTrain, [], param);  % (501*1200, 1*1200)
        % ---O:1/I:501/E:1/Nt:1/Ns:1200/Ti:200(200)/
        % + Iter =  200, M =   95, err = 0.818802, F = -2.82021, H = 123.268

        % 在测试集中根据图像预测特征
        yPredPercept = predict_output(xTestPercept, model, param)';  % 1750x1 <- 501x1750
        yPredImagery = predict_output(xTestImagery, model, param)';  % 500x1  <- 501x500

        predictPercept = [predictPercept, yPredPercept];
        predictImagery = [predictImagery, yPredImagery];
    end

    %% 对每个类别的预测做平均
    categoryTestPercept = unique(floor(testPerceptLabels));
    categoryTestImagery = unique(floor(testimageryLabels));

    predictPerceptAveraged = [];
    predictImageryAveraged = [];
    for j = 1:length(categoryTestPercept)
        categ = categoryTestPercept(j);
        predictPerceptAveraged(j, :) = mean(predictPercept(floor(testPerceptLabels) == categ, :));
    end
    for j = 1:length(categoryTestImagery)
        categ = categoryTestImagery(j);
        predictImageryAveraged(j, :) = mean(predictImagery(floor(testimageryLabels) == categ, :));
    end

    %% 保存数据
    [rDir, rFileBase, rExt] = fileparts(resultFile);
    setupdir(rDir);

    save(resultFile, ...
         'predictPercept', 'predictImagery', ...
         'predictPerceptAveraged', 'predictImageryAveraged', ...
         'categoryTestPercept', 'categoryTestImagery', ...
         '-v7.3');

    %% 移除文件锁
    unlockcomput(analysisId, lockDir);

end

fprintf('%s done\n', mfilename);
