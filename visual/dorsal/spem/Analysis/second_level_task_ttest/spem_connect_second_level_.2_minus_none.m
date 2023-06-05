%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% second-level t-test between .2 Hz smooth pursuit and fixation
%-----------------------------------------------------------------------
matlabbatch{1}.spm.stats.factorial_design.dir = {'your_results_path'};                % please customize to your individual results path
%%
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = {'your_first_level_vector'}; % please customize to your vector of first level con_0001 contrast paths
                                                        
%%
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

%%
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;


%% run batch
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, '');

%% save batch

cd(char(matlabbatch{1}.spm.stats.factorial_design.dir))
batch_name = ['con_0001' '.mat'];
save (batch_name, 'matlabbatch');

