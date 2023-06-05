
cur_dir = fileparts( mfilename('fullpath') );
project_home_dir = fileparts( fileparts(cur_dir) );

% 运行整个项目的配置文件
run(fullfile(project_home_dir, 'config.m'));


%% 添加Path
utils_dir = fullfile(project_home_dir, 'utils');
addpath(utils_dir);

% add path for extract_all_file
addpath(fullfile(project_home_dir, 'gaze'));

% add path for 'spm12' library
addpath(fullfile(project_home_dir, 'utils', 'spm12'));

% add path for 'spm12' library
addpath(fullfile(project_home_dir, 'utils', 'xjview'));

% add path for 'matlab_utils' library
addpath(fullfile(project_home_dir, 'utils', 'matlab_utils'));
