% 获取凝视时的注意力的范围大小，用圆圈的大小表示，输出到目录: data/seg/{sub_id}/track/gaze_scope.txt
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
            fprintf('Process seg %d\n', s);
            cur_seg = cur_segs(s);  cur_seg = cur_seg{1};
            seq_idx = seq_idx+1;
            
            % 存放每一帧所对应的平均x,y,r
            mean_physio = zeros(numel(fieldnames(cur_seg)) ,4);
            % 帧号--受试--x--y--瞳孔半径
            % 行数：该跟踪序列的帧数 × 受试者数
            mean_physio_accum = zeros(numel(fieldnames(cur_seg)) * numel(all_subs), 5);
            accum_idx = 0;
            % 考虑多个受试，求凝视点的平均
            for j = 1 : length(all_subs)
                physio_path = fullfile(data_home_dir, all_subs{j}, 'ses-movie/func', ...
                    sprintf('%s_ses-movie_task-movie_run-%d_recording-eyegaze_physio.tsv', ...
                    all_subs{j}, run_ids));
                cur_physio_data = importdata(physio_path);  % 获得当前受试者的物理数据（眼动记录数据）
                for seg_idx = 1 : numel(fieldnames(cur_seg))
                    frame_id = str2num( eval(['cur_seg.f_' num2str(seg_idx)]) );
                    cur_frame_data = cur_physio_data(cur_physio_data(:, 4) == frame_id, :);  % 获取当前帧的所有眼动数据
                    accum_idx = accum_idx+1;
                    mean_physio_accum(accum_idx, 1) = seg_idx;  % 相对帧号（从1开始）
                    mean_physio_accum(accum_idx, 2) = j;        % 相对受试编号（从1开始，间隔为1）
                    cur_mean = mean(cur_frame_data);            % 对但钱帧的所有眼动数据求平均
                    mean_physio_accum(accum_idx, 3) = cur_mean(1);  % X
                    mean_physio_accum(accum_idx, 4) = cur_mean(2);  % Y
                    mean_physio_accum(accum_idx, 5) = cur_mean(3);  % pupil radius
                end
            end
            
            % 去除包含NaN的行
            mean_physio_accum(isnan(mean_physio_accum(:,3)),:) = [];  %删除a矩阵中第3列（同时第4列也是NaN）包含NaN的行
            for seg_idx = 1 : numel(fieldnames(cur_seg))
                cur_frame_physio = mean_physio_accum(mean_physio_accum(:, 1) == seg_idx, :);
                mean_physio(seg_idx, 1) = seg_idx;
                cur_seg_mean = mean(cur_frame_physio);
                mean_physio(seg_idx, 2) = cur_seg_mean(3);  % X
                mean_physio(seg_idx, 3) = cur_seg_mean(4);  % Y
                mean_physio(seg_idx, 4) = cur_seg_mean(5);  % pupile radius
            end
            
            % 保存结果
            gaze_scope_path = fullfile(proj_home_dir, 'data', 'seg', mat2str(seq_idx), 'track', 'gaze_scope.txt');
            save(gaze_scope_path, 'mean_physio', '-ascii');
        end
    end
end

% /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv
% 暂时只计算受试者1的眼动信息
