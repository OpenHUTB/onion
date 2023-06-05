% createfigure    Create figures for generic decoding results
% 绘制图1：看到的 和 看到/想象的 平均类别 特征解码精度
% 图2：看到的和想象的类别识别精度
%
% Author: Tomoyasu Horikawa <horikawa-t@atr.jp>, Shuntaro C. Aoki <aoki@atr.jp>
%

clear;
cur_dir = fileparts(mfilename('fullpath'));
run(fullfile(cur_dir, 'init.m'));

%% 数据设置
% resultsDir = './results/';
resultsFileFeatPred = fullfile(resultsDir, 'FeaturePrediction.mat');
resultsFileCatIdent = fullfile(resultsDir, 'CategoryIdentification.mat');

%% 图像设置
fontSize = 5;
lineWidth = 2;
subplotMargin = 0.13;

figureProperties = {'Color', 'white'};
axesProperties = {'Box', 'off', ...
                  'TickDir', 'out'};

%% 加载结果
resFeatPred = load(resultsFileFeatPred);
resCatIdent = load(resultsFileCatIdent);

subjectList = unique({resFeatPred.results(:).subject});
%roiList     = unique({resFeatPred.results(:).roi});
roiList     = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};
featureList = unique({resFeatPred.results(:).feature});

%% Realign results to a 3D array (subject x ROI x feature)
results = [];
for i = 1:length(subjectList)
for j = 1:length(roiList)
for k = 1:length(featureList)
    fInd = strcmp({resFeatPred.results(:).subject}, subjectList{i}) ...
           & strcmp({resFeatPred.results(:).roi}, roiList{j}) ...
           & strcmp({resFeatPred.results(:).feature}, featureList{k});
    cInd = strcmp({resCatIdent.results(:).subject}, subjectList{i}) ...
           & strcmp({resCatIdent.results(:).roi}, roiList{j}) ...
           & strcmp({resCatIdent.results(:).feature}, featureList{k});

    results.featPred.image.perception(i, j, k)    = resFeatPred.results(fInd).predaccImagePercept;
    results.featPred.category.perception(i, j, k) = resFeatPred.results(fInd).predaccCategoryPercept;
    results.featPred.category.imagery(i, j, k)    = resFeatPred.results(fInd).predaccCategoryImagery;

    results.catIdent.perception(i, j, k) = resCatIdent.results(cInd).correctRatePerceptAve;
    results.catIdent.imagery(i, j, k)    = resCatIdent.results(cInd).correctRateImageryAve;
end
end
end

%% 可视化结果：特征解码精度
%  看图像的特征 & 想像类别平均特征解码精度

dataType = {'seen:image', 'seen:category', 'imagined:category'};

% 图形设置
numRow = 14;
numCol = 6;
add = 3;
[plotOrder, numRow, numCol] = get_subplot_order([numRow, numCol], 'lbu', [3, 0]);

% 可视化结果
hf = makefigure('fullscreen');
set(hf, figureProperties{:});

cnt = 0;
for iData = 1:length(dataType)

    numSbj = size(results.featPred.image.perception, 1);
    
    switch dataType{iData}
      case 'seen:image'
        col = cmap4('bg4');
        dat = results.featPred.image.perception;
        range = [-0.2, 0.6];
        yax = -0.2:0.2:0.6;
      case 'seen:category'
        col = cmap4('bg4');
        dat = results.featPred.category.perception;
        range = [-0.2,0.6];
        yax = -0.2:0.2:0.6;
      case 'imagined:category'
        col = cmap4('ibg4');
        dat = results.featPred.category.imagery;
        range = [-0.2, 0.4];
        yax = -0.2:0.2:0.4;
    end

    % 计算平均和置信度区间
    mu = squeeze(mean(dat, 1));
    ci = tinv(0.95, numSbj - 1) .* squeeze(std(dat, [], 1)) ./ sqrt(numSbj);
    
    for ix = 1:length(featureList)
        cnt = 1 + cnt;
        plotIndex = cnt + (iData - 1) * add;
        ha = subplottight(numRow, numCol, plotOrder(plotIndex), subplotMargin);
        set(ha, 'FontSize', fontSize);
        hold on;

        % 数据绘制
        bar(ha, mu(:, ix), ...
            'facecolor', col{1}, ...
            'edgecolor', 'none', ...
            'LineWidth', lineWidth);
        errorbar_h(ha, mu(:, ix), ci(:, ix), '.k');

        % 水平线绘制
        hline(yax, '-k');

        % 文本绘制
        text(1, -0.1, ...
             sprintf('%s; %s', dataType{iData}, featureList{ix}), ...
             'FontSize', fontSize);

        % x and y axis
        % Draw axis labels only on plots at the bottom of the figure
        if ix == 1 || ix == 9
            draw_axes_label(roiList, 1);
        else
            set(ha, 'XTickLabel', '');
        end
        xlim([0.5, length(roiList) + 0.5]);

        ylabel('Corr. coeff.');
        draw_axes_label(yax, 2, yax);
        ylim(range);

        % 设置坐标轴参数
        set(ha, axesProperties{:});
    end
end

suptitle(sprintf('Seen image feature and seen/imagined category-average feature decoding accuracy'));

savefigure(hf, fullfile(resultsDir, 'FeaturePredictionAccuracy.pdf'));

%% 可视化结果：识别精度

dataType = {'seen', 'imagined'};

% 图形设置
numRow = 14;
numCol = 6;
add = 3;
[plotOrder, numRow, numCol] = get_subplot_order([numRow, numCol], 'lbu', [3, 1]);

% 可视化结果
hf = makefigure('fullscreen');
set(hf, figureProperties{:});

cnt = 0;
for iData = 1:length(dataType)

    switch dataType{iData}
        case 'seen'
            col = cmap4('bg4');
            dat = results.catIdent.perception;
        case 'imagined'
            col = cmap4('ibg4');
            dat = results.catIdent.imagery;
    end

    % 计算平均和置信度区间
    mu = squeeze(mean(dat, 1));
    ci = tinv(0.95, numSbj - 1) .* squeeze(std(dat, [], 1)) ./ sqrt(numSbj);
    
    for ix = 1:length(featureList)
        cnt = 1 + cnt;
        plotIndex = cnt + (iData - 1) * add;
        ha = subplottight(numRow, numCol, plotOrder(plotIndex), subplotMargin);
        set(ha, 'FontSize', fontSize);
        hold on;

        % 数据绘制
        bar(ha, mu(:, ix) * 100, ...
            'facecolor', col{1}, ...
            'edgecolor', 'none', ...
            'LineWidth', lineWidth);
        errorbar_h(ha, mu(:, ix) * 100, ci(:, ix) * 100, '.k');

        % 水平线绘制
        hline(50, '-k')
        hl = hline(60:10:100, '-');
        set(hl, 'Color', col{1});

        % 文本额绘制
        text(1, 95, ...
             sprintf('%s: %s', dataType{iData}, featureList{ix}), ...
             'FontSize', fontSize);

        % x 和 y 轴
        if ix == 1 || ix == 9
            draw_axes_label(roiList, 1);
        else
            set(ha, 'XTickLabel', '');
        end
        xlim([0.5, length(roiList) + 0.5]);

        ylabel('Accuracy (%)');
        ylim([40, 100]);

        
        % 设置坐标轴参数
        set(ha, axesProperties{:});
    end
end % 图像类别

suptitle(sprintf('Seen and imagined category identification accuracy'));

savefigure(hf, fullfile(resultsDir, 'IdentificationAccuracy.pdf'));
