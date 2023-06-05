clc,close,clear

run('../config.m');

sub_num = 36;
subjects_list = [1]; % [1,2]
segments_num = 8;  % The 7 parts are concatenated and the split again into eight segments


%% Load gaze data
for i = 1 : length(subjects_list)
    disp(subjects_list(i));
    func_path = fullfile(data_home_dir, ...
        sprintf('sub-%02d', subjects_list(i)), ...
        'ses-movie', ...
        'func'...
    );

    gaze_infos = {};
    for j = 1 : segments_num
        % sub-01_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv
        gaze_dir = fullfile(func_path, ...
            sprintf('sub-%02d_ses-movie_task-movie_run-%d_recording-eyegaze_physio.tsv', ...
            subjects_list(i), j));
        disp(fprintf('load %s', gaze_dir));
        gaze_info = dlmread(gaze_dir);  % read ASCII-delimited file of numeric data into matrix
        gaze_infos = [gaze_infos, gaze_info];
    end
    
    
    % display current gaze video (1280x546)
    cur_frame = imresize(imread('ngc6543a.jpg'), [HEIGHT, WIDTH]);
end


%% Load fMRI data
