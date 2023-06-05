
%% 环境初始化
run('./config.m');
proj_dir = fileparts( mfilename('fullpath') );

%% 视频数据处理

%% 视觉跟踪测试
track_forrest_cmd = [python2_path, ' ', ...
    fullfile(proj_dir, 'scripts', 'demo_btn_forrest.py')];
disp(track_forrest_cmd);
[~, track_out] = system(track_forrest_cmd);
% 使用命令行运行Python脚本和Pycharm运行Python有区别
% 当前工作路径应修改为工程主目录，而不是脚本所在的目录。


%% 处理眼动数据
gaze_analysis_dir = fullfile(proj_dir, 'gaze', 'analysis');
addpath(gaze_analysis_dir);

% 将“研究阿甘”的眼动数据转换为Arff格式
% Studyforrest2ArffRegex(... 
%     fullfile(data_home_dir, '*', '*', '*', '*movie*events.tsv'), ...
%     fullfile(data_home_dir, '*', '*', '*', '*eyegaze_physio.tsv'), ...
%     fullfile(proj_dir, 'result' ,'gaze_data') ...
% );

% 平稳跟踪检测
% utils/sp_tools/run_detection.py

% 回归出平稳跟踪、注视和运动
sp_sacc_motion_analysis_dir = fullfile(gaze_analysis_dir, 'sp_saccade_motion_analysis');
addpath(sp_sacc_motion_analysis_dir);
% ComputeRegressorsAll.m


%% 处理核磁共振数据

% 对所有受试者视频段1到8进行预处理
% PreprocessRegex('/data2/whd/workspace/data/neuro/software/ds000113/sub-*/')
PreprocessRegex(fullfile(data_home_dir, 'sub-*/'));


% 一阶段分析（个体层次）
% run_1stLevelAnalysis.m

% 不同眼动类型之间对比分析
% run_ContrastAnalysis.m

% 二阶段（不同受试者之间）分析并可视化
score_dir = fullfile(proj_dir, 'score');
addpath(score_dir);
% analysis_fMRI.m

%% 视觉跟踪和平稳跟踪之间的相似度分数

% 获得和平稳跟踪相关脑区的镜像数据
% get_image_data.m

% 计算视觉跟踪和平稳跟踪的相似度分数
% track_score.m


%% Latex编译
cd latex
!pdflatex -synctex=1 -interaction=nonstopmode "btn".tex
cd ..
