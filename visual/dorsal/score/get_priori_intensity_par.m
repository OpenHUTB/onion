%% 从先验知识的脑区划分中得到强度值

%%
clc;
addpath('utils');
proj_home = fileparts( fileparts( mfilename('fullpath') ) );
res_dir = fullfile(proj_home, 'result');
addpath(fullfile(proj_home, 'utils', 'matlab_utils'));  % 配置glob.m函数


%% Load Brain Area Data
X = load('data/TDdatabase');
wholeMaskMNIAll = X.wholeMaskMNIAll;

V1_mask = wholeMaskMNIAll.brodmann_area_17;
MT_mask = wholeMaskMNIAll.brodmann_area_21;
FEF_mask = wholeMaskMNIAll.brodmann_area_8;

% pcshow(pointCloud(V1_mask))
% pcshow(pointCloud(mni))


%% 从图像帧所对应大脑所有位置（立体空间）的强度值（80*80*35*1）中获得指定脑区的强度值
% data/generate_segs.m 
% 所加载立体空间中强度值的坐标如何对应到MNI空间的坐标，（如何定义80*80*35*1中的坐标原点？）

% 从矩阵坐标转换至MNI坐标
% 假设矩阵坐标：X=3，Y = 5, Z = 6;
% 方法：mx = 90-3*(X-1); my = 3*(Y-1)-126; mz = 3*(Z-1) - 72;
seg_dir = fullfile(proj_home, 'data', 'seg', '*');
dirs = glob(seg_dir);


if isempty(gcp('nocreate'))
    parpool(8);
end
for dir = dirs'
    cur_dir = dir{1};
    fprintf('Process directory %s\n', cur_dir);
    func_path = fullfile(cur_dir, 'func', '*.mat');
    funcs = glob(func_path);
    for sub = 1 : length(funcs)  % 暂时只处理一个，后面可以添加成多个受试
        fprintf('Process subject %d.\n', sub);
        cur_func = funcs{sub};
        cur_seg_vols = load(cur_func);  
        [~, cur_sub] = fileparts(cur_func);

        intensity_total = cur_seg_vols.cur_seg_vols;
        [x_size, y_size, z_size, n_frame] = size(intensity_total);

        V1_intensity_accum = cell(n_frame, 1);
        % 保留强度不为0的体素
        parfor f = 1 : n_frame
            fprintf('frame %d\n', f);
            cur_intensity = intensity_total(:, :, :, f);

            valid_idx = cur_intensity > 0;
            valid_intensity = cur_intensity(valid_idx);  % 每个有效位置的强度值N*1
            valid_mni = zeros(length(valid_intensity), 3);  % 每个有效位置的(x,y,z)坐标: N*3
            mni_idx = 0;
            for i = 1 : size(valid_idx, 1)
                for j = 1 : size(valid_idx, 2)
                    for k = 1 : size(valid_idx, 3)
                        if valid_idx(i, j, k) > 0
                            mni_idx = mni_idx + 1;
                            % 从矩阵坐标转换至MNI坐标（正确性？）
                            valid_mni(mni_idx, 1) = 90 - 3*(i-1);
                            valid_mni(mni_idx, 2) = 3*(j-1) - 126;
                            valid_mni(mni_idx, 3) = 3*(k-1) - 76;
                        end
                    end
                end
            end
            mni = valid_mni;
            intensity = valid_intensity;

            %% 按照制定脑区进行强度值过滤 Filter with brain area
            V1_intensity = zeros(1, length(V1_mask));
            MT_intensity = zeros(1, length(MT_mask));
            FEF_intensity = zeros(1, length(FEF_mask));
            V1_idx = 0;
            MT_idx = 0;
            FEF_idx = 0;
            
            % 获得V1脑区的激活强度值
            for i = 1 : size(V1_mask, 1)  % 对V1脑区的每一个体素从整个皮层激活中寻找，包括则保存下激活强度值
                idx = row_index(mni, V1_mask(i, :));  % V1掩模的三维坐标去匹配 标准模板mni中的坐标是否存在，返回行索引
                if idx > 0
                    V1_intensity(i) = intensity(idx);
                end
            end
            fprintf('Frame %d, V1 intensity got\n', f);

            % 获得MT脑区的激活强度值
            for i = 1 : size(MT_mask, 1)  % 对V1脑区的每一个体素从整个皮层激活中寻找，包括则保存下激活强度值
                idx = row_index(mni, MT_mask(i, :));  % V1掩模的三维坐标去匹配 标准模板mni中的坐标是否存在，返回行索引
                if idx > 0
                    MT_intensity(i) = intensity(idx);
                end
            end
            fprintf('Frame %d, MT intensity got\n', f);

            % 获得FEF脑区的激活强度值
            for i = 1 : size(FEF_mask, 1)  % 对V1脑区的每一个体素从整个皮层激活中寻找，包括则保存下激活强度值
                idx = row_index(mni, FEF_mask(i, :));  % V1掩模的三维坐标去匹配 标准模板mni中的坐标是否存在，返回行索引
                if idx > 0
                    FEF_intensity(i) = intensity(idx);
                end
            end
            fprintf('Frame %d, FEF intensity got\n', f);
            

            fprintf('V1 point number: %d\n', sum(V1_intensity ~= 0));  % 25
            fprintf('MT point number: %d\n', sum(MT_intensity ~= 0));
            fprintf('FEF point number: %d\n', sum(FEF_intensity ~= 0));

            % V1 point number: 25
            % MT point number: 70
            % FEF point number: 70

            % 为parfor循环外保存结果做准备
            V1_intensity_accum{f} = V1_intensity;
            MT_intensity_accum{f} = MT_intensity;
            FEF_intensity_accum{f} = FEF_intensity;
        end
        
        %% Save Result
        feature_dir = fullfile(cur_dir, 'feature');      % 存放所有深度学习模型跟踪视频片段每一帧的特征
        intensity_dir = fullfile(cur_dir, 'intensity');  % 存放所有受试观察视频片段每一帧时指定脑区的强度值
        if ~exist(feature_dir, 'dir'), mkdir(feature_dir); end
        if ~exist(intensity_dir, 'dir'), mkdir(intensity_dir); end

        % 保存受试者sub-01在视频帧frame-1上脑区V1、MT、FEF上的激活强度
        for f = 1 : n_frame
            cur_V1_intensity = V1_intensity_accum{f};
            cur_MT_intensity = MT_intensity_accum{f};
            cur_FEF_intensity = FEF_intensity_accum{f};
            save(fullfile(intensity_dir, sprintf('%s_frame-%d_V1_intensity.txt', cur_sub, f)), 'cur_V1_intensity', '-ascii');
            save(fullfile(intensity_dir, sprintf('%s_frame-%d_MT_intensity.txt', cur_sub, f)), 'cur_MT_intensity', '-ascii');
            save(fullfile(intensity_dir, sprintf('%s_frame-%d_FEF_intensity.txt', cur_sub, f)), 'cur_FEF_intensity', '-ascii');
        end

    end
end

delete(gcp('nocreate'));

% 使用镜像文件做测试，后面用指定图像的大脑强度值代替
% image_filename = '/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-3_bold.nii';
% image_filename = '/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/sub-01/spmT_0001.nii';
% [imageFileName, M, DIM, TF, df, mni, intensity] ...
%     = get_image_file(image_filename);


