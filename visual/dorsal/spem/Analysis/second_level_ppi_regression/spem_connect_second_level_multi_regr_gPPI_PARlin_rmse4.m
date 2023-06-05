%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% second-level regression: gPPI starting in left PPC and RMSE for .4 Hz smooth pursuit 
%-----------------------------------------------------------------------
matlabbatch{1}.spm.stats.factorial_design.dir = {'your_results_path'};                % please customize to your individual results path
%%
matlabbatch{1}.spm.stats.factorial_design.des.mreg.scans = {'your_first_level_vector'}; % please customize to your vector of first level con_PPI_SPEM .4_minus_none contrast paths
                                                                                        % double-check: order has to be the same as in matlabbatch{1}.spm.stats.factorial_design.cov.c;


matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov = struct('c', {}, 'cname', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.des.mreg.incint = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.c = [93.86
97.81
106.93
101.25
93.08
100.93
100.91
103.09
96.19
68.7
105.19
106.79
86.54
56.92
140.32
69.54
57.67
69.38
67.49
78.38
46.57
57.56
136.25
44.21
58.46
42.54
90.16
96.19
51.58
71.38
44.24
67.31
57.36
61.8
65.93
88.24
60.1
52.77
53.96
54.2
57.05
52.52
];
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'rmse_04';
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
batch_name = 'rmse_4';
save (batch_name, 'matlabbatch');

%% run batch
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, '');

