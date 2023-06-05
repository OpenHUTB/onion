% analysis_FeaturePredictionAccuracy
% 计算特征预测的精度

cur_dir = fileparts(mfilename('fullpath'));
run(fullfile(cur_dir, 'init.m'));

%% 数据设置
% subjectList  : 受试 IDs （元胞数组）
subjectList = {'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'};
% featureList  : 图像特征列表[元胞数组]
featureList = {'cnn1', 'cnn2', 'cnn3', 'cnn4', ...
               'cnn5', 'cnn6', 'cnn7', 'cnn8', ...
               'hmax1', 'hmax2', 'hmax3', 'gist', 'sift'};
% roiList      : 感兴趣区域列表[元胞数组]
roiList     = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};

% Image feature data
imageFeatureFile = 'ImageFeatures.mat';

%% File name settings
predResultFileNameFormat = @(s, r, f) fullfile(resultsDir, sprintf('%s/%s/%s.mat', s, r, f));
resultFile = fullfile(resultsDir, 'FeaturePrediction.mat');
if exist(resultFile, 'file'); delete(resultFile); end


%% Analysis Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('%s started\n', mfilename);

%%----------------------------------------------------------------------
%% Initialization
%%----------------------------------------------------------------------

addpath(genpath('./lib'));

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end


%% Load data

%% 加载图像特征
fprintf('Loading image feature data...\n');

[feat.dataSet, feat.metaData] = load_data(fullfile(dataDir, imageFeatureFile));


%% 创建分析参数举证 (analysisParam)

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


%% Analysis loop

results = [];

for n = 1:size(analysisParam, 1)

    %% Initialization --------------------------------------------------

    % Get data index in the current analysis
    iSbj = analysisParam(n, 1);
    iRoi = analysisParam(n, 2);
    iFeat = analysisParam(n, 3);

    % Set analysis ID and a result file name
    analysisId = sprintf('%s-%s-%s-%s', ...
                         mfilename, ...
                         subjectList{iSbj}, ...
                         roiList{iRoi}, ...
                         featureList{iFeat});

    % Check or double-running
    if checkfiles(resultFile)
        fprintf('The analysis is already done and skipped\n');
        continue;
    end

    fprintf('Start %s\n', analysisId);

    %% 获得图像特征
    layerFeat = select_feature(feat.dataSet, feat.metaData, sprintf('%s = 1', featureList{iFeat}));
    catIds = get_dataset(feat.dataSet, feat.metaData, 'CatID');
    featType = get_dataset(feat.dataSet, feat.metaData, 'FeatureType');

    %% 聚集特征单元
    predResultFile = predResultFileNameFormat(subjectList{iSbj}, roiList{iRoi}, featureList{iFeat});
    % code/matlab/results/Subject1/V1/cnn1.mat'. No such file or directory
    % 原因：code/matlab/tmp中存在cnn1.mat被锁住的文件
    res = load(predResultFile);  

    predPercept = res.predictPerceptAveraged;
    predImagery = res.predictImageryAveraged;

    categoryPercept = res.categoryTestPercept;
    categoryImagery = res.categoryTestImagery;

    %% 计算类别特征预测精度

    %% 获得测试特征（图像）
    testFeat = layerFeat(featType == 2, :);
    testCatIds = catIds(featType == 2, :);

    testFeat = get_refdata(testFeat, testCatIds, categoryPercept);

    %% 图像特征预测精度（概要关联）(profile correlation)
    predAcc.image.perception = nanmean(diag(fastcorr(predPercept, testFeat)));  % mean(A,'omitnan')

    %% 获得测试特征（类别平均）
    catTestFeat = layerFeat(featType == 3, :);
    catTestCatIds = catIds(featType == 3, :);

    catTestFeatPercept = get_refdata(catTestFeat, catTestCatIds, categoryPercept);
    catTestFeatImagery = get_refdata(catTestFeat, catTestCatIds, categoryImagery);

    %% 类别平均特征预测精度 (profile correlation)
    predAcc.category.perception = nanmean(diag(fastcorr(predPercept, catTestFeatPercept)));
    predAcc.category.imagery    = nanmean(diag(fastcorr(predImagery, catTestFeatImagery)));

    results(n).subject = subjectList{iSbj};
    results(n).roi = roiList{iRoi};
    results(n).feature = featureList{iFeat};
    results(n).categoryTestPercept = categoryPercept;
    results(n).categoryTestImagery = categoryImagery;
    results(n).predictPercept = predPercept;
    results(n).predictImagery = predImagery;
    results(n).predaccImagePercept = predAcc.image.perception;
    results(n).predaccCategoryPercept = predAcc.category.perception;
    results(n).predaccCategoryImagery = predAcc.category.imagery;
end

%% 保存数据
save(resultFile, 'results', '-v7.3');

fprintf('%s done\n', mfilename);
