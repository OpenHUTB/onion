%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% second-level regression: gPPI starting in right V1 and saccadic frequendy for .2 Hz smooth pursuit 
%-----------------------------------------------------------------------
matlabbatch{1}.spm.stats.factorial_design.dir = {'your_results_path'};                % please customize to your individual results path
%%
matlabbatch{1}.spm.stats.factorial_design.des.mreg.scans = {'your_first_level_vector'}; % please customize to your vector of first level con_PPI_SPEM .2_minus_none contrast paths
                                                                                        % double-check: order has to be the same as in matlabbatch{1}.spm.stats.factorial_design.cov.c;


matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov = struct('c', {}, 'cname', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.des.mreg.incint = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.c = [0.51
0.31
0.03
0.25
0.2
0.8
0.37
0.37
0.54
0.61
0.47
0.34
0.35
0.5
1.47
0.33
0.21
0.06
0.45
0.74
1.42
0.76
1.4
1.03
0.99
0.75
0.35
1.01
0.12
0.88
0.37
0.78
0.39
0.94
0.59
0.1
0.04
0.24
0.16
0.05
0.49
0.21
];
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'sacfreq_02';
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
batch_name = 'sacfreq_2';
save (batch_name, 'matlabbatch');

%% run batch
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, '');

