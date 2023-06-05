%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% second-level regression: gPPI starting in right V5 and saccadic frequency for .4 Hz smooth pursuit 
%-----------------------------------------------------------------------
matlabbatch{1}.spm.stats.factorial_design.dir = {'your_results_path'};                % please customize to your individual results path
%%
matlabbatch{1}.spm.stats.factorial_design.des.mreg.scans = {'your_first_level_vector'}; % please customize to your vector of first level con_PPI_SPEM .4_minus_none contrast paths
                                                                                        % double-check: order has to be the same as in matlabbatch{1}.spm.stats.factorial_design.cov.c;

matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov = struct('c', {}, 'cname', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.des.mreg.incint = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.c = [1.52
0.62
0.41
0.56
0.97
1.49
1.05
0.86
1.21
2.28
0.71
1.56
1.33
1.19
2.57
1.36
0.95
0.37
1.42
1.24
2.47
1.64
2.19
2.26
1.86
1.38
1.21
1.54
0.35
1.81
1.07
2.1
1.59
2.19
0.88
1.95
0.3
0.67
1.12
0.36
1.63
1.73
];
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'sacfreq_04';
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
batch_name = 'sacfreq_4';
save (batch_name, 'matlabbatch');

%% run batch
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, '');

