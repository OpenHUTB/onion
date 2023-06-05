%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% second-level regression: gPPI starting in right V5 and pursuit gain for .2 Hz smooth pursuit 
%-----------------------------------------------------------------------
matlabbatch{1}.spm.stats.factorial_design.dir = {'your_results_path'};                % please customize to your individual results path
%%
matlabbatch{1}.spm.stats.factorial_design.des.mreg.scans = {'your_first_level_vector'}; % please customize to your vector of first level con_PPI_SPEM .2_minus_none contrast paths
                                                                                        % double-check: order has to be the same as in matlabbatch{1}.spm.stats.factorial_design.cov.c;

matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov = struct('c', {}, 'cname', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.des.mreg.incint = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.c = [0.94
0.98
0.9
1.01
0.88
1.02
0.91
0.92
1
0.76
1.03
0.89
0.93
0.89
0.67
0.99
0.92
0.93
1
0.86
0.9
0.89
0.93
0.91
0.93
0.96
0.87
1.04
1.01
0.95
0.93
0.92
0.98
1
1.01
0.92
0.97
1.03
0.96
0.95
1
0.89];
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'gain_02';
matlabbatch{1}.spm.stats.factorial_design.cov.iCFI = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.iCC = 1;
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;


%% save batch

cd(char(matlabbatch{1}.spm.stats.factorial_design.dir))
batch_name = 'gain_2';
save (batch_name, 'matlabbatch');

%% run batch
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, '');

