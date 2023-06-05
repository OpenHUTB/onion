
% 加载跟踪序列图像的详细帧号信息
run('../config.m');
%%
movie_segs_path = fullfile(proj_home_dir, 'data', 'movie_segs.json');
movie_segs_strs = importdata(movie_segs_path);    % 将.json文件导入成字符串组成的cell数组（每行1个cell）
movie_segs_str = join(string(movie_segs_strs));   % cell数组 -> string数组 -> 合并数组
% 当前用于跟踪的视频片段的帧号、sub-01_ses-movie_task-movie_run-7_events.tsv文件中的行号
movie_segs = jsondecode(char(movie_segs_str));    % string -> char数组 -> json解析


all_fields = fieldnames(movie_segs);

seq_idx = 0;
seq_accum = {};

for i = 1 : length(all_fields)
    cur_field = all_fields(i);  cur_field = cur_field{1};  % 获得跟踪段的名字，比如seg_5
    
    if startsWith(cur_field, 'seg_')  % 过滤掉json中不以seg_开头的数据
        cur_field_slit = split(cur_field, '_');
        run_ids = cur_field_slit{2};  run_ids = str2num(run_ids);
        cur_segs = eval(['movie_segs.' cur_field]);
        if isstruct(cur_segs)  % 表示从当前视频段中只切分出一个跟踪段
            cur_segs = {cur_segs};
        end
        for s = 1 : length(cur_segs)  % 当前跟踪段的id(1,2,3,4...)
            cur_seg = cur_segs(s);  cur_seg = cur_seg{1};
            seq_idx = seq_idx+1;
            seq_accum{seq_idx} = cur_seg;
%             cur_all_fileds = fieldnames(cur_seg);
        end
    end
end