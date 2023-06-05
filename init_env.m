
project_home_dir = fileparts(fileparts( mfilename('fullpath') ));

% 运行整个项目的配置文件
% run(fullfile(project_home_dir, 'config.m'));


%% 添加环境变量
utils_dir = fullfile(project_home_dir, 'utils');
addpath(utils_dir);

% add path for extract_all_file
% addpath(fullfile(project_home_dir, 'gaze'));

% add path for 'spm12' library
utils_dir = fullfile(matlabroot, 'software', 'matlab_utils');
addpath(fullfile(utils_dir, 'spm12'));
addpath(fullfile(utils_dir, 'spm12', 'matlabbatch'));

% add path for 'spm12' library
addpath(fullfile(utils_dir, 'xjview'));

addpath(fullfile(utils_dir))

% add path for 'matlab_utils' library
% addpath(fullfile(project_home_dir, 'utils', 'matlab_utils'));
