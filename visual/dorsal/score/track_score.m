%%
% 分别加载script/demo_btn_forrest_batch.py得到的深度模块vt得到的激活特征 和score/get_priori_intensity_par.m处理得到的大脑激活强度信息

clc; clear;

feature_dim = 1000;
gaze_radius_scale = 8;  % 半径缩小的倍数

cur_dir = fileparts( mfilename('fullpath') );
% project_home_dir = fileparts(cur_dir);

run('../config.m');
% addpath(fullfile(project_home_dir, 'utils', 'matlab_utils'));


%% LSTM feature
seg_dir = fullfile(proj_home_dir, 'data', 'seg/');
segs = dir(seg_dir);
segs = segs(cellfun(@valid_dir_name, {segs.name}));
segs_name = cellfun(@(x) str2double(x), {segs.name});  % 对cell数组中的名字字段都转换成int，以便进行排序
[blah, order] = sort(segs_name);
segs = segs(order);  % 对cell数组按照排序好的名字字段进行重新调整行顺序

vt_feature_accum = cell(length(segs), 1);
MT_intensity_accum = cell(length(segs), 1);
iou = 0;
for i = 1 : length(segs)
    cur_seg_dir = fullfile(segs(i).folder, segs(i).name);
    img_num = numel(dir(fullfile(cur_seg_dir, '*.png')));  % 获得当前跟踪序列的图片数
    vt_feature = importdata(fullfile(cur_seg_dir, 'feature',  ...
        sprintf('frame-1-%02d_vt_feature', img_num) ) );  % 每个跟踪序列都少了一张图片的vt特征？
    vt_feature_accum{i} = vt_feature;
    
    intensity_dir = fullfile(cur_seg_dir, 'intensity');
    MT_idx = 0;
    cur_MT_intensity = [];
    for j = 2 : size(vt_feature, 1)+1
        MT_idx = MT_idx+1;
        cur_MT_intensity(MT_idx, :) = importdata(fullfile(intensity_dir, ...
            sprintf('sub-01_frame-%d_MT_intensity.txt', j)));
    end
    MT_intensity_accum{i} = cur_MT_intensity;
    
    %% 计算目标跟踪框和凝视范围的交兵比
    % 加载目标跟踪框的结果
    pred_bbox_path = fullfile(cur_seg_dir, 'track', 'pred_bbox.txt');
    pred_bbox = importdata(pred_bbox_path);
    % 加载凝视范围的平均结果
    gaze_scope_path = fullfile(cur_seg_dir, 'track', 'gaze_scope.txt');
    gaze_scope = importdata(gaze_scope_path);
    
    % 验证框和凝视范围是否正确
    intersection_num = 0;
    union_num = 0;
    imgs = glob(fullfile(cur_seg_dir, '*.png'));
    
    if is_debug && i==6 && 0
        for img_idx = 1 : length(imgs)
            cur_img = imread(imgs{img_idx});
            if i == 6 && img_idx == 2
                imshow(cur_img);
                rectangle('position', ...
                    [pred_bbox(img_idx, 2), pred_bbox(img_idx, 1), pred_bbox(img_idx, 3), pred_bbox(img_idx, 4)], ...
                    'EdgeColor','cyan', 'LineWidth', 2)
                saveas(gcf, fullfile(proj_home_dir, 'latex', 'imgs', ['track_' num2str(i) '_' num2str(img_idx) '_rectangle.png']));
                % draw circle
                imshow(cur_img);
                h = drawcircle('Center',[gaze_scope(img_idx, 2),gaze_scope(img_idx,3)], ...
                    'Radius',gaze_scope(img_idx,4)/gaze_radius_scale, ...
                    'Color','r');
                saveas(gcf, fullfile(proj_home_dir, 'latex', 'imgs', ['track_' num2str(i) '_' num2str(img_idx) '_circle.png']));
            else
                imshow(cur_img);
                rectangle('position', ...
                    [pred_bbox(img_idx, 2), pred_bbox(img_idx, 1), pred_bbox(img_idx, 3), pred_bbox(img_idx, 4)], ...
                    'EdgeColor','cyan', 'LineWidth', 2)
                h = drawcircle('Center',[gaze_scope(img_idx, 2),gaze_scope(img_idx,3)], ...
                    'Radius',gaze_scope(img_idx,4)/gaze_radius_scale, ...
                    'Color','r');
            end
            
            if i == 10 && img_idx == 4 || i==8 && img_idx==2 || i == 2 && img_idx ==7 || i == 4 && img_idx == 2 || i == 6 && img_idx == 2
                saveas(gcf, fullfile(proj_home_dir, 'latex', 'imgs', ['track_' num2str(i) '_' num2str(img_idx) '.png']));
            end
        end
    end
    
    % 计算长方形的目标跟踪框和圆形的凝视范围之间的交并比（感兴趣区域掩模求交集的方式）
    for img_idx = 1 : length(imgs)
        cur_img = imread(imgs{img_idx});
        % 原来数据保存到额是：[垂直, 水平, 高, 宽]
        x = pred_bbox(img_idx, 2);
        y = pred_bbox(img_idx, 1);
        w = pred_bbox(img_idx, 3);
        h = pred_bbox(img_idx, 4);

        xi = [x, x+w, x+w, x];
        yi = [y, y, y+h, y+h];
        rect_BW = roipoly(cur_img, xi, yi);

        % 圆形边界的坐标
        cir_x = gaze_scope(img_idx, 2);
        cir_y = gaze_scope(img_idx,3);
        cir_r = gaze_scope(img_idx,4)/gaze_radius_scale;
        sita=0:0.05:2*pi;
        cir_xi = cir_x+cir_r*cos(sita);
        cir_yi = cir_y + cir_r * sin(sita);
        cir_BW = roipoly(cur_img, cir_xi, cir_yi);

        intersection_num = intersection_num + sum(sum(and(rect_BW, cir_BW)));  % 'all'
        union_num = union_num + sum(sum(or(rect_BW, cir_BW)));
            
    end
    
    iou = iou + intersection_num / union_num;
end
iou = iou / length(segs);
fprintf('IOU similarity: %f\n', iou);


%% 使用PCA进行数据降维
warning('off', 'stats:pca:ColRankDefX')  % Warning: Columns of X are linearly dependent to within machine precis

% num_comp = 10;          % PCA压缩到的维度
vt_feature_accum = cell2mat(vt_feature_accum);
MT_intensity_accum = cell2mat(MT_intensity_accum);
nums_comp = 3 : 1 : 137;  % < 137
comp_sim = zeros(size(nums_comp, 2), 2);  % 保存成分数和相似度之间的关系
sim_idx = 0;
for num_comp = nums_comp  % 分析各种PCA的成分数目（PCA压缩比例）对皮尔逊相关系数的影响
    % ref: https://www.pianshen.com/article/4427148264/
    [vt_pc, vt_score, vt_latent, vt_tsquare] = pca(vt_feature_accum', 'NumComponents', num_comp); %feature是799*216的矩阵; 'NumComponents', 6
    [MT_pc, MT_score, MT_latent, MT_tsquare] = pca(MT_intensity_accum', 'NumComponents', num_comp);

    % 用latent来计算降维后取多少维度能够达到自己需要的精度
    % fprintf('vt precision using PCA: %.4f.\n', cumsum(vt_latent)./sum(vt_latent));
    % 一般取到高于95%就可以了，这里我们取前40维，精度达到了0.9924

    % pca函数已经给出了所有的转换后矩阵表示,也就是输出的score项，取出前40维就是降维后特征
    % feature_after_PCA = score(:,1:5);

    % Pearson correlation coefficient
    % corr 使用 Student t 分布来转换相关性以计算 Pearson 相关性的 p 值。
    % 当X和Y来自正态分布时，这种相关性是精确的。
    % 当两矩阵某两列相关性较大，rho对应行、列的值较大。
    %  p 值小于显著性水平 0.05，这表明拒绝两列之间不存在相关性的假设
    [rho, pval] = corr(vt_pc, MT_pc, 'Type', 'Pearson');

    % 理想情况下，结果矩阵中，需要rho中对应位置的值较大，pval的值较小。
    % 实现：显著性水平小于0.05的情况下，rho中相关性最大，即为所求的分数
    % Q: 没有考虑其他列
    req_pos = pval < 0.1;
    satisfied_rho = abs(rho.*req_pos);
    % max(reshape(satisfied_rho, 1, numel(satisfied_rho)))
    [max_satisfied, row_idx] = max(satisfied_rho);
    [max_sim, col_idx] = max(max_satisfied);
    row_idx = row_idx(int32(col_idx));
    max_p = pval(row_idx, col_idx);
    fprintf('max fMRI similarity: %f\nmax p value: %f\n', max_sim, max_p);
    
    sim_idx = sim_idx + 1;
    comp_sim(sim_idx, 1) = num_comp;
    comp_sim(sim_idx, 2) = max_sim;
    comp_sim(sim_idx, 3) = max_p;
    
    %% p < 0.01 和p < 0.05没区别
%     req_pos = pval < 0.01;
%     satisfied_rho = abs(rho.*req_pos);
%     [max_satisfied, row_idx] = max(satisfied_rho);
%     [max_sim, col_idx] = max(max_satisfied);
%     row_idx = row_idx(int32(col_idx));
%     max_p = pval(row_idx, col_idx);
%     comp_sim(sim_idx, 4) = max_sim;
%     comp_sim(sim_idx, 5) = max_p;
end

% 归一化max_sim列和max_p列
for i = 2 : size(comp_sim, 2)
    min_val = min(comp_sim(:, i));
    max_val = max(comp_sim(:, i));
    ratio_val = 1 / (max_val - min_val);
    comp_sim(:, i) = (comp_sim(:, i) - min_val)*ratio_val;
end

% 绘制PCA成分数与皮尔逊相关系数之间关系的图
save('../result/tmp/comp_sim.mat', 'comp_sim');
ts_plot(comp_sim);


%% 计算 fMRI激活 和 行为相似度 的平均值
overall_sim = (max_sim + iou) / 2;
fprintf('Overall similarity: %f\n', overall_sim);


%% 结果
% All IoU: 0.401056
% IOU similarity: 0.113398
% max fMRI similarity: 0.341036
% max p value: 0.000045
% Overall similarity: 0.227217

% IOU similarity: 0.081494
% max fMRI similarity: 0.305663
% max p value: 0.000281
% Overall similarity: 0.193579

% IOU similarity: 0.057081
% max fMRI similarity: 0.406842
% max p value: 0.000001
% Overall similarity: 0.231962

% 初始加入IOU相似度
% IOU similarity: 0.043329
% max fMRI similarity: 0.406842
% max p value: 0.000001
% Overall similarity: 0.225085

% 修正初始框
% max similarity: 0.406842
% max p value: 0.000001

% 第一次跑通
% max similarity: 0.351979
% max p value: 0.000025





