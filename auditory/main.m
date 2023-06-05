%% 类脑音频识别
%% 准备音频数据

run('music_genre_fMRI\process_music.mlx');
%% 训练音频分类网络

warning off
run('genre_classification.mlx');
warning on

%% 大脑激活数据分析

% TODO
% spm
run('music_genre_fMRI/preprocess/main.m')
%% 提取对应音频的大脑激活和深度模型激活

run('get_fMRI_activation.mlx');
%% 大脑激活和模型激活的回归和预测

run('analysis_FeaturePrediction'); % 得出类脑相似性分数

%% 论文生成
% 图片生成
% 下载eps2pdf工具
utils_rep = 'https://github.com/OpenHUTB/utils';
git_path = fullfile(matlabroot, 'software', 'git', 'bin', 'git.exe');
utils_dir = fullfile("C:", 'workspace', 'utils'); % workspace
if ~exist(utils_dir, 'dir')
    clone_cmd =  append(git_path, " clone ", utils_dir);
    system(clone_cmd);
    % todo
end
%%
addpath(fullfile(utils_dir, 'export_fig'));
% 如果出现需要定位：Ghostcript not found. Please locate the program.
% 则下载并安装Ghostcript: 
% https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10011/gs10011w64.exe
eps2pdf(fullfile(cur_dir, 'latex', 'authors', 'HaidongWang.eps'), ...
    fullfile(cur_dir, 'latex', 'authors', 'HaidongWang.pdf'));


% latex编译论文
cc
run('../../utils/init_matlab.mlx');
cc
cd('latex');
latex_cmd = [fullfile(latex_exe_dir, 'pdflatex.exe') ' -synctex=1 -interaction=nonstopmode ban.tex'];
system(latex_cmd);
winopen('ban.pdf')
% init_robot
%% 
% 测试


% BUTTON1_MASK（鼠标左键），BUTTON2_MASK（鼠标中键）；BUTTON3_MASK（鼠标右键）
robot.keyPress    (java.awt.event.KeyEvent.VK_CONTROL);
robot.mousePress  (java.awt.event.InputEvent.BUTTON1_MASK);
robot.mouseRelease(java.awt.event.InputEvent.BUTTON1_MASK);
robot.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL);

% robot.mousePress(KeyEvent.VK_CONTROL);
% robot.mousePress(KeyEvent.VK_2);
% robot.mouseRelease(KeyEvent.VK_CONTROL);
% robot.mouseRelease(KeyEvent.VK_2);