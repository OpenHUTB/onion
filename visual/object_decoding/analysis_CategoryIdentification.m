% analysis_CategoryIdentification    运行目标类别识别

%% 目录设置
cur_dir = fileparts(mfilename('fullpath'));
run(fullfile(cur_dir, 'init.m'));


%% 初始化设置 

% 数据设置
% subjectList  : List of subject IDs [cell array]
% featureList  : List of image features [cell array]
% roiList      : List of RoiList [cell array]

subjectList = {'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'};
featureList = {'cnn1', 'cnn2', 'cnn3', 'cnn4', ...
               'cnn5', 'cnn6', 'cnn7', 'cnn8', ...
               'hmax1', 'hmax2', 'hmax3', 'gist', 'sift'};
roiList     = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};

% 图像特征数据
imageFeatureFile = 'ImageFeatures.mat';


%% 文件名设置
predResultFile = fullfile(resultsDir, 'FeaturePrediction.mat');
resultFile = fullfile(resultsDir, 'CategoryIdentification.mat');


%% 分析主程序

fprintf('%s started\n', mfilename);


%% 初始化

addpath(genpath('./lib'));

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end;


%% 加载数据

% 加载图像特征
fprintf('Loading image feature data...\n');

[feat.dataSet, feat.metaData] = load_data(fullfile(dataDir, imageFeatureFile));

%% 创建分析参数矩阵 (analysisParam)
% 5个受试 * 10个roi * 13个特征层
analysisParam = uint16(zeros(length(subjectList) * length(roiList) * length(featureList), 3)); 

c = 1;
for iSbj = 1:length(subjectList)
for iRoi = 1:length(roiList)
for iFeat = 1:length(featureList)
    analysisParam(c, :) = [ iSbj, iRoi, iFeat ];
    c =  c + 1;
end
end
end


%% 分析循环
featpred = load(predResultFile);

results = [];
for n = 1:size(analysisParam, 1)

    %% 初始化

    % 在当前分析中获得数据索引
    iSbj = analysisParam(n, 1);
    iRoi = analysisParam(n, 2);
    iFeat = analysisParam(n, 3);

    % 设置分析ID和结果文件名
    analysisId = sprintf('%s-%s-%s-%s', ...
                         mfilename, ...
                         subjectList{iSbj}, ...
                         roiList{iRoi}, ...
                         featureList{iFeat});

    % Check or double-running
    if checkfiles(resultFile)
        % 分析结果已经存在就跳过
        fprintf('The analysis is already done and skipped\n');
        continue;
    end

    fprintf('Start %s\n', analysisId);

    %% 获得图像特征
    layerFeat = select_feature(feat.dataSet, feat.metaData, ...
                              sprintf('%s = 1', featureList{iFeat}));
    catIds = get_dataset(feat.dataSet, feat.metaData, 'CatID');
    featType = get_dataset(feat.dataSet, feat.metaData, 'FeatureType');

    %% 加载数据 (unit_all) --------------------------------------------
    ind = strcmp({featpred.results(:).subject}, subjectList{iSbj}) ...
          & strcmp({featpred.results(:).roi}, roiList{iRoi}) ...
          & strcmp({featpred.results(:).feature}, featureList{iFeat});

    % TODO: 增加索引检查

    predPercept = featpred.results(ind).predictPercept;
    predImagine = featpred.results(ind).predictImagery;

    categoryPercept = featpred.results(ind).categoryTestPercept;

    %% 目标类别识别分析

    %% 获得类别特征
    featCatTest = layerFeat(featType == 3, :);
    catIdsCatTest = catIds(featType == 3, :);

    featCatTest = get_refdata(featCatTest, catIdsCatTest, categoryPercept);

    featCatOther = layerFeat(featType == 4, :);
    catIdCatOther = catIds(featType == 4, :);

    %% 配对识别
    labels = 1:size(featCatTest, 1);
    candidate = [featCatTest; featCatOther(1:50, :)];  % 测试类别特征；其他类别特征  

    % 所见到的类别
    simmat = fastcorr(predPercept', candidate');
%     simmat = fastcorr(predPercept, featCatTest); % simmat = fastcorr(predPercept', candidate');
    correctRate.perception = pwidentification(simmat, labels);  % 两个输入数组的非单一维度必须相互匹配。

    % 所想到的类别
    simmat = fastcorr(predImagine', candidate');
    correctRate.imagery = pwidentification(simmat, labels);

    fprintf('Correct rate (seen)     = %.f%%\n', mean(correctRate.perception) * 100);
    fprintf('Correct rate (imagined) = %.f%%\n', mean(correctRate.imagery) * 100);

    results(n).subject = subjectList{iSbj};
    results(n).roi = roiList{iRoi};
    results(n).feature = featureList{iFeat};
    results(n).correctRatePercept = correctRate.perception;
    results(n).correctRateImagery = correctRate.imagery;
    results(n).correctRatePerceptAve = mean(correctRate.perception);
    results(n).correctRateImageryAve = mean(correctRate.imagery);
end

%% 保存数据
save(resultFile, 'results', '-v7.3');

fprintf('%s done\n', mfilename);
