%% 从先验知识的脑区划分中得到听觉处理的强度值

%% 初始化
clc; clear;
run('../init.m');
run('../../init_env.m')

addpath('utils');
proj_home = fileparts(fileparts(fileparts( fileparts( mfilename('fullpath') ) )));
res_dir = fullfile(proj_home, 'data', 'brain', 'visual', 'dorsal', 'result');
addpath(fullfile(proj_home, 'brain', 'utils', 'matlab_utils'));  % 配置glob.m函数

%% 展示数据
% 使用 xjview 打开模型推理所生成的 T统计量（是根据模型得到的，用来做检验的统计量）图像。
% 正常T-statistic应该在0假设(null hypothesis)为真时，服从T分布(T-distribution)。
% T-test是根据T-statistic值的大小计算p-value，决定是接受还是拒绝假设。
% xjview(fullfile(proj_dir, 'brain', 'auditory', 'music_genre_fMRI', 'preprocess', 'ds003720-download', 'sub-001', 'func', 'classical', 'spmT_0001.nii'));

% pcshow(pointCloud(V1_mask))
% pcshow(pointCloud(mni))


%% 从图像帧所对应大脑所有位置（立体空间）的强度值（80*80*35*1）中获得指定脑区的强度值
% data/generate_segs.m 
% 所加载立体空间中强度值的坐标如何对应到MNI空间的坐标，（如何定义80*80*35*1中的坐标原点？）

% 从矩阵坐标转换至MNI坐标
% 假设矩阵坐标：X=3，Y = 5, Z = 6;
% 方法：mx = 90-3*(X-1); my = 3*(Y-1)-126; mz = 3*(Z-1) - 72;

sub_dirs = glob(fullfile(proj_home, 'brain', 'auditory', ...
    'music_genre_fMRI' ,'preprocess', 'ds003720-download', 'sub-*'));

%% 提取fMRI的ROI
% 将fMRI数据提取到 `dataFileList` 变量中的Subject1.mat等文件中
for dir = sub_dirs'
    cur_dir = dir{1};
    fprintf('Process directory %s\n', cur_dir);
    func_path = fullfile(cur_dir, 'func', 'swrsub*.nii');
    funcs = glob(func_path);
    for run_id = 1 : length(funcs)
        fprintf('Process run ID %d.\n', run_id);
        cur_func = funcs{run_id};
        [~, cur_func_name, ~] = fileparts(cur_func);
        
        cur_vol_infs = spm_vol(cur_func);        % 必须先读取volums的信息然后再使用spm_read_vols进行数据的读取
        cur_vols = spm_read_vols(cur_vol_infs);  % 读取当前镜像的所有激活数据
        [x_size, y_size, z_size, n_frame] = size(cur_vols);

        % 保留强度不为0的体素（三维空间激活特征如何与深度网络激活进行对比）
        for f = 1 : n_frame
            fprintf('frame %d\n', f);
            cur_intensity = cur_vols(:, :, :, f);

%             mni_idx = 0;
%             key_set = cell(length(cur_intensity),1);
%             value_set = zeros(length(cur_intensity), 1);
%             for i = 1 : size(cur_intensity, 1)
%                 for j = 1 : size(cur_intensity, 2)
%                     for k = 1 : size(cur_intensity, 3)
%                         mni_idx = mni_idx + 1;
%                         % 从矩阵坐标转换至MNI坐标（正确性？）https://zhuanlan.zhihu.com/p/466707281
% %                         valid_mni(mni_idx, 1) = 90 - 3*(i-1);
% %                         valid_mni(mni_idx, 2) = 3*(j-1) - 126;
% %                         valid_mni(mni_idx, 3) = 3*(k-1) - 76;
%                         cur_x = 90 - 3*(i-1);
%                         cur_y = 3*(j-1) - 126;
%                         cur_z = 3*(k-1) - 72;  % 76
%                         cur_key = sprintf('%d,%d,%d', cur_x, cur_y, cur_z);
%                         key_set{mni_idx} = cur_key;
%                         value_set(mni_idx) = cur_intensity(i, j, k);
% %                         valid_mni(mni_idx, 4) = cur_intensity(i, j, k);
%                     end
%                 end
%             end
% %             mni_intensity = valid_mni;
%             % 构建坐标和强度值的HashMap
%             pos_intensity_map = containers.Map(key_set, value_set);
            

            %% 按照制定脑区进行强度值过滤 Filter with brain area
            A1_intensity = zeros(1, length(A1_mask));
            A2_intensity = zeros(1, length(A2_mask));
            A3_intensity = zeros(1, length(A3_mask));
            T2_T3_intensity = zeros(1, length(T2_T3_mask));            

            % 获得A1脑区的激活强度值
            for i = 1 : size(A1_mask, 1)
                cur_pos = A1_mask(i,:);
                cur_i = ceil((90-cur_pos(1))/3)+1;
                cur_j = ceil((cur_pos(2)+126)/3) + 1;
                cur_z = ceil((cur_pos(3))/3) + 1;
                A1_intensity(i) = cur_intensity(cur_i, cur_j, cur_z);
            end

            % 获得A2脑区的激活强度值
            for i = 1 : size(A2_mask, 1)
                cur_pos = A2_mask(i,:);
                cur_i = ceil((90-cur_pos(1))/3)+1;
                cur_j = ceil((cur_pos(2)+126)/3) + 1;
                cur_z = ceil((cur_pos(3))/3) + 1;
                A2_intensity(i) = cur_intensity(cur_i, cur_j, cur_z);
            end

            % 获得A3脑区的激活强度值
            for i = 1 : size(A3_mask, 1)
                cur_pos = A3_mask(i,:);
                cur_i = ceil((90-cur_pos(1))/3)+1;
                cur_j = ceil((cur_pos(2)+126)/3) + 1;
                cur_z = ceil((cur_pos(3))/3) + 1;  if (cur_z < 1); cur_z = 1; end
                A3_intensity(i) = cur_intensity(cur_i, cur_j, cur_z);
            end

            % 获得T2_T3脑区的激活强度值
            for i = 1 : size(T2_T3_mask, 1)
                cur_pos = T2_T3_mask(i,:);
                cur_i = ceil((90-cur_pos(1))/3)+1;
                cur_j = ceil((cur_pos(2)+126)/3) + 1;
                cur_z = ceil((cur_pos(3))/3) + 1;  if (cur_z < 1); cur_z = 1; end
                T2_T3_intensity(i) = cur_intensity(cur_i, cur_j, cur_z);
            end

            %% Save Result
            feature_dir = fullfile(cur_dir, 'feature');      % 存放所有深度学习模型跟踪视频片段每一帧的特征
            intensity_dir = fullfile(cur_dir, 'intensity');  % 存放所有受试观察视频片段每一帧时指定脑区的强度值
            if ~exist(feature_dir, 'dir'), mkdir(feature_dir); end
            if ~exist(intensity_dir, 'dir'), mkdir(intensity_dir); end

            % 保存受试者sub-01在Run-x视频帧frame-1上脑区A1、A1、A3上的激活强度
            % 为什么中间有很多0（在计算相似性的时候过滤掉？）
            save(fullfile(intensity_dir, sprintf('%s_frame-%d_A1_intensity.txt', cur_func_name, f)), ...
                'A1_intensity', '-ascii');
            save(fullfile(intensity_dir, sprintf('%s_frame-%d_A2_intensity.txt', cur_func_name, f)), ...
                'A2_intensity', '-ascii');
            save(fullfile(intensity_dir, sprintf('%s_frame-%d_A3_intensity.txt', cur_func_name, f)), ...
                'A3_intensity', '-ascii');
            save(fullfile(intensity_dir, sprintf('%s_frame-%d_T2_T3_intensity.txt', cur_func_name, f)), ...
                'T2_T3_intensity', '-ascii');
        end
        
    end
end

%% 



% 使用镜像文件做测试，后面用指定图像的大脑强度值代替
% image_filename = '/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-3_bold.nii';
% image_filename = '/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/sub-01/spmT_0001.nii';
% [imageFileName, M, DIM, TF, df, mni, intensity] ...
%     = get_image_file(image_filename);


