
% addpath('/data2/whd/workspace/sot/hart/utils');
run('../gaze/analysis/init_env.m');

cur_dir = fileparts( mfilename('fullpath') );
project_home_dir = fileparts(cur_dir);

addpath( fullfile( project_home_dir, 'gaze', 'analysis') );
PreprocessRegex('/data2/whd/workspace/data/neuro/software/ds000113/sub-*/')

addpath( fullfile( project_home_dir, 'gaze', 'analysis', 'sp_saccade_motion_analysis') );


