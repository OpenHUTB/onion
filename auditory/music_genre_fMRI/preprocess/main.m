
clear;clc;
run('../../../init_env.m')
run('../../init')


%% 下载原始fMRI文件
if ~exist(fMRI_raw_path, 'file')
    % error;
    fprintf('Please download raw fMRI from <a href = "/openneuro.org/datasets/ds003720/versions/1.0.0">Music Genre fMRI dataset</a>\n');
    % exit; % exit和quit都会退出matlab
    return;
else
    if ~exist(fMRI_dir, 'dir')
        unzip(fMRI_raw_path, fMRI_dir);
    end
%     untar(fMRI_raw_path, fMRI_dir);
end

% Unrecognized function or variable 'cfg_branch'.
% enter spm/matlabbatch directory, add path
PreprocessRegex(fullfile(fMRI_dir, 'sub-*/'));




% Processing run 14 from /tmp/ds003720-download/sub-005/
% 
% 
% ------------------------------------------------------------------------
% 29-Aug-2022 11:22:46 - Running job #1
% ------------------------------------------------------------------------
% 29-Aug-2022 11:22:46 - Running 'Realign: Estimate & Reslice'
% 
% SPM12: spm_realign (v7141)                         11:22:46 - 29/08/2022
% ========================================================================
% Completed                               :          11:24:32 - 29/08/2022
% 
% SPM12: spm_reslice (v7141)                         11:24:32 - 29/08/2022
% ========================================================================
% 29-Aug-2022 11:24:57 - Failed  'Realign: Estimate & Reslice'
% 错误使用 mat2file
% Problem writing data (could be a disk space or quota issue).
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/@file_array/subsasgn.m" (v7147), function "subfun" at line 164.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/@file_array/subsasgn.m" (v7147), function "subsasgn" at line 85.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/spm_write_plane.m" (v6079), function "spm_write_plane" at line 31.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/spm_write_vol.m" (v5731), function "spm_write_vol" at line 84.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/spm_reslice.m" (v7141), function "reslice_images" at line 248.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/spm_reslice.m" (v7141), function "spm_reslice" at line 136.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/config/spm_run_realign.m" (v7141), function "spm_run_realign" at line 44.
% 
% 29-Aug-2022 11:24:58 - Running 'Coregister: Estimate'
% 
% SPM12: spm_coreg (v7320)                           11:24:58 - 29/08/2022
% ========================================================================
% 29-Aug-2022 11:24:58 - Failed  'Coregister: Estimate'
% 错误使用 spm_vol>spm_vol_hdr
% File "/tmp/ds003720-download/sub-005/func/meansub-005_task-Test_run-02_bold.nii" does not exist.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/spm_vol.m" (v5958), function "spm_vol_hdr" at line 80.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/spm_vol.m" (v5958), function "spm_vol" at line 61.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/spm_coreg.m" (v7320), function "spm_coreg" at line 117.
% In file "/home/ubuntu/software/matlab_2022a_linux/utils/spm12/config/spm_run_coreg.m" (v5956), function "spm_run_coreg" at line 22.
% 
% 29-Aug-2022 11:24:58 - Running 'Segment'
% 
% SPM12: spm_preproc_run (v7670)                     11:24:58 - 29/08/2022
% ========================================================================
% Segment /tmp/ds003720-download/sub-005/anat/sub-005_T1w.nii,1
% Completed                               :          11:29:51 - 29/08/2022
% 29-Aug-2022 11:29:51 - Done    'Segment'
% The following modules did not run:
% Failed: Realign: Estimate & Reslice
% Failed: Coregister: Estimate
% 
% 错误使用 MATLABbatch system
% Job execution failed. The full log of this run can be found in MATLAB command
% window, starting with the lines (look for the line showing the exact #job as
% displayed in this error message)
% ------------------
% Running job #1
% ------------------
%  
% 保存命令历史记录时遇到错误
% 保存命令历史记录时遇到错误
% 桌面配置保存失败