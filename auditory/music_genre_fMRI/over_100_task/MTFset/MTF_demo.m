
addpath('/data3/whd/workspace/brain_model/auditory/music_genre_fMRI/over_100_task/MTFset');
addpath('/data3/whd/workspace/brain_model/utils/gammatonegram');

file_name = 'Run1_1.wav';
file_dir = '/data3/whd/workspace/result/auditory/genres_forExp/';
file_length = 15;  % 音乐的长度15s
% The filter output averaged across 1.5 s (TR) was used as a feature in the cochlear model
TR = 1.5;
MTFCode = 24;  % 程序推荐使用（和论文一致）
MakeFeatureSpace_MTF(file_name, file_dir, file_length, TR, MTFCode)
