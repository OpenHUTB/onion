% nohup matlab -nodesktop -nosplash -r extract_all_file > extract_all_file.log &
function r = extract_all_file(data_home_dir)
% clc,close,clear

% diary('extract_all_file.log');
% diary on;

% cur_dir = fileparts( mfilename('fullpath') );
% eval(sprintf('cd %s', cur_dir));
% addpath(cur_dir);

% run('../config.m');
% addpath('../utils/');

% only extract the files in specified directory by overwriting this
% variable.
% data_home_dir = '/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/'; % sub-03

%% 
% sub-01/ses-forrestgump/func
% the processed data is copied from '/data2/whd/workspace/data/neuro/ds000113-download/'
% data_home_dir = '/data2/whd/workspace/data/neuro/software/ds000113/';
% all_files = RangTraversal(data_home_dir);

% extract file to current directory
i = 0;
FileData = {};

all_files = RangTraversal(data_home_dir); % 获取所有文件
len_File = size(all_files); % 获取文件列表长度

for i = 1:len_File
    cur_file = all_files{i};
    dot = max(find(cur_file == '.'));
    suffix = cur_file(dot:dot+2);
	% 后缀为.gz文件
	if strcmp(suffix, '.gz') == 1
        disp(cur_file);
        [cur_file_path, cur_file_name] = fileparts(cur_file);
        
        gunzip(cur_file, cur_file_path);
        
		i = i+1;
	end
end

% diary off;

end
