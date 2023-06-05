
is_debug = true;
close all;

python2_path = '/home/d/anaconda2/envs/hart/bin/python';
AI_proj_dir = '/data2/whd/workspace/AI';
if isunix
    data_prefix = '/data2/whd/workspace/data/neuro/';
else
    % 在windows操作系统上映射到linux的网络驱动上
    data_prefix = 'Z:/data2/whd/workspace/data/neuro/';
end
raw_data_dir = fullfile(data_prefix, 'ds000113-download');

% 整个项目的主目录
proj_home_dir = fileparts( mfilename('fullpath') );
data_home_dir = fullfile(data_prefix, 'software', 'ds000113');

% Path
addpath(fullfile(proj_home_dir, 'utils'));
addpath(fullfile(proj_home_dir, 'utils', 'matlab_utils'));
% addpath(fullfile(proj_home_dir, 'score', 'utils'));
addpath(fullfile(AI_proj_dir, 'tools', 'matlab'));

% full video: 1280*720
WIDTH = 1280;
HEIGHT = 720;  % 2.3443 (2.35855)
FPS = 25;

%% 核磁共振参数
TR = 2;  % 每隔2秒进行一次扫描，电影图片的抽取也是一样

%% 电影数据
movie_dir = fullfile(data_prefix, 'movie');

%% 所考虑的所有受试
subs_dir = glob(fullfile(data_home_dir, 'sub-*'));
all_subs = {};
for s = 1 : length(subs_dir)
    [~, cur_sub , ~] = fileparts( fileparts(subs_dir{s}));
    all_subs{s} = cur_sub;
end
clear subs_dir s cur_sub
disp('config finished');



