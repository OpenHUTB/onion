
addpath('over_100_task/MTFset');
addpath('../../utils/gammatonegram');

%% 
file_name = 'Run1_1.wav';
% result_dir = '/data3/whd/workspace/result/auditory';
if ispc
    workspace_dir = 'C:/dong/data/brain/auditory/process_music_fMRI';
else
    workspace_dir = '/home/ubuntu/dong/data/brain/auditory/process_music_fMRI';
end

run_dir = fullfile(workspace_dir, 'genres_forExp/');
wave15s_dir = fullfile(workspace_dir, 'genres_wav15s/');

file_length = 15;  % 音乐的长度15s
% The filter output averaged across 1.5 s (TR) was used as a feature in the cochlear model
TR = 1.5;
MTFCode = 24;  % 程序推荐使用（和论文一致）

% 总共720个15秒音乐片段（包括训练集12和测试集6）：40×(12+6)
train_order = load(fullfile(run_dir, 'ExpTrnOrder.mat'), '-mat', 'TrnOrd');
train_order = train_order.TrnOrd;
val_order = load(fullfile(run_dir, 'ExpValOrder.mat'));
val_order = val_order.ValOrd;

% 抽取训练集和验证集所有刺激的特征
all_stimulus = [train_order, val_order];
if ~exist(fullfile(workspace_dir, 'MTF_features.mat'), 'file')
    MTF_features = zeros(size(all_stimulus, 1), size(all_stimulus, 2), 2000, 10);
    for i = 1 : size(all_stimulus, 1)
        for j = 1 : size(all_stimulus, 2)
            cur_stimulus = all_stimulus(i,j); file_name = cur_stimulus{1};
            cur_MTF_feature = MakeFeatureSpace_MTF(file_name, wave15s_dir, file_length, TR, MTFCode);  % 2000*10
            MTF_features(i, j, :, :) = cur_MTF_feature;
            disp(fprintf('Processed (%d,%d): %s\n', i, j, file_name));
        end
    end
    save(fullfile(workspace_dir, 'MTF_features.mat'), "MTF_features", '-v7.3');
else
    load(fullfile(workspace_dir, 'MTF_features.mat'))
end




%% 使用PCA进行特征压缩
% 将MTF特征从2000维压缩成302维
% MTF_features(2000*7200)的行对应于观测值，列对应于变量（每个音乐刺激对应10列，总共720个刺激）。
% 数据矩阵MTF_features的主成分系数coeff: 302*7200
% 40*18*2000*10
[coeff, score, latent, tsquared, explained, mu] = pca(squeeze(MTF_features(1,1,:,:)));  % , 'NumComponents', 302

% 重构数据(在执行PCA之前减去均值mu，重构时需要将其重新添加)
% reconstructed = score * coeff' + repmat(mu, 2000, 1);



