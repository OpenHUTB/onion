% function PreprocessRegex.m
% 对所有的受试数据进行处理（暂时还没有GLM处理的过程）
%
% 对所有受试者的视频段1到8进行预处理
% This function finds all the directories in the provided regular expression
% and preprocesses the data for runIds from 1 to 8
%
% input:
%   regex   -   dir regex
%
% ex. PreprocessRegex('/mnt/scratch/ioannis/studyforrest_preprocessed/sub-*/')
% 
% 在 Matlab 命令行中运行：
% PreprocessRegex('/data3/dong/brain/auditory/music_genre_fMRI/preprocess/ds003720-download/sub-*/')

function PreprocessRegex(regex)
    
    dirs = glob(regex);  % dirs = {dirs{2:5,1}}' % for debug

    runIds = 1:18;
    % test data
%     runIds = 18 : -1 : 1;

    for dir=dirs'
        % prepare data
        dst_dir = fullfile(dir{1}, 'func');
        
        % 重置测试镜像文件
%         copyfile('/data3/whd/data/neuro/music_genre/ds003720-download_bak/sub-001/func/sub-001_task-Training_run-01_bold.nii', ...
%             dst_dir);
        
%         if exist(dst_dir, 'dir')
%            rmdir(dst_dir, 's'); 
%         end
%         path_names = strsplit(dir{1}, '/');
%         cur_subject = path_names(end-1);
%         src_dir = fullfile(raw_data_dir, cur_subject, 'ses-movie', 'func');
%         copyfile(src_dir{1}, dst_dir);
%         extract_all_file(dst_dir);
        
        for runId = runIds
            fprintf('Processing run %d from %s\n', runId, dir{1});
            Preprocess(dir{1}, runId);  
%             spm_model(dir{1}, runId);
        end
    end
end


%% 
% SPM12: spm_coreg (v7320)                           00:22:35 - 22/03/2022
% ========================================================================
% 22-Mar-2022 00:22:35 - Failed  'Coregister: Estimate'
% Error using spm_vol>spm_vol_hdr (line 80)
% File "/data3/dong/brain/auditory/music_genre_fMRI/preprocess/ds003720-download/sub-002/anat/sub-001_T1w.nii" does not exist.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/spm_vol.m" (v5958), function "spm_vol_hdr" at line 80.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/spm_vol.m" (v5958), function "spm_vol" at line 61.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/spm_coreg.m" (v7320), function "spm_coreg" at line 123.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/config/spm_run_coreg.m" (v5956), function "spm_run_coreg" at line 22.
% 
% 22-Mar-2022 00:22:35 - Running 'Segment'
% 
% SPM12: spm_preproc_run (v7670)                     00:22:35 - 22/03/2022
% ========================================================================
% Segment /data3/dong/brain/auditory/music_genre_fMRI/preprocess/ds003720-download/sub-002/anat/sub-001_T1w.nii,1
% 22-Mar-2022 00:22:36 - Failed  'Segment'
% Error using spm_vol>spm_vol_hdr (line 80)
% File "/data3/dong/brain/auditory/music_genre_fMRI/preprocess/ds003720-download/sub-002/anat/sub-001_T1w.nii" does not exist.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/spm_vol.m" (v5958), function "spm_vol_hdr" at line 80.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/spm_vol.m" (v5958), function "spm_vol" at line 61.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/spm_preproc_run.m" (v7670), function "run_job" at line 86.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/spm_preproc_run.m" (v7670), function "spm_preproc_run" at line 41.
% In file "/data2/whd/workspace/sot/hart/utils/spm12/config/spm_cfg_preproc8.m" (v7629), function "spm_local_preproc_run" at line 474.
% 
% The following modules did not run:
% Failed: Coregister: Estimate
% Failed: Segment
% 
% Error using MATLABbatch system
% Job execution failed. The full log of this run can be found in MATLAB command window, starting with the lines (look for the line showing the
% exact #job as displayed in this error message)
% ------------------
% Running job #1
% ------------------



%%
% '/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/meansub-01_ses-movie_task-movie_run-1_bold.nii'
% is empty

% ds000113/sub-01/anatomy/highres001.nii" does not exist.
% T1-weighted image
% Location sub<ID>/anatomy/highres001.nii.gz
% locate: ds000113/sub-01/ses-forrestgump/anat/sub-01_ses-forrestgump_T1w.nii
