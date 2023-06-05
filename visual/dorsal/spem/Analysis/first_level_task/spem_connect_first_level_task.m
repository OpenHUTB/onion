%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% first-level analysis of task effects
%-----------------------------------------------------------------------
%% load general project settings

subjects_path = 'your_project_path'; % please customize to path, where 
                                     % preprocessed data is saved (one folder per participant)
                                     % each of these individual folders
                                     % should contain a folder called
                                     % "SPEM" where the preprocessed data
                                     % (starting with "swcr") and 
                                     % realignment parameters are saved
subjects = list_vp_names(subjects_path); 


%% loop preprocessing script through all participants

%% prepare variables

for k=1:length(subjects)

subject_name = subjects{k}; % change back to curly brackets

fprintf('Processing subject "%s", "%s" \n' ,subject_name);

subject_path = fullfile(subjects_path, subject_name);
cd(subject_path);
functional_path = fullfile(subject_path, 'SPEM');
smooth_data = cellstr(spm_select('FPList',functional_path, '^swcr.*\.img$'));
smooth_data = smooth_data; % selects all smoothed functional images 
smooth_data = strcat(smooth_data , ',1');
nonparametric_dir = fullfile(functional_path, 'nonparametric');
realignment_parameters = cellstr(spm_select('FPList',functional_path, '^rp.*\.txt')) ; 
%% fill batch

matlabbatch{1}.spm.stats.fmri_spec.dir = {nonparametric_dir}; 
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2.5;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 1;
matlabbatch{1}.spm.stats.fmri_spec.sess.scans = cellstr(smooth_data); 
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'SPEM .2';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = [0.28
                                                         188.31
                                                         314.31
                                                         503.32
                                                         566.33];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 30;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 1;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'SPEM .4';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = [62.3116875
                                                         125.3073125
                                                         251.314125
                                                         377.3205625
                                                         440.3211875];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 30;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 1;
matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = realignment_parameters; 
matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'SPEM .2 > Fix';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 0];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'SPEM .4 > Fix';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [0 1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{3}.tcon.name = 'SPEM .2 > SPEM .4';
matlabbatch{3}.spm.stats.con.consess{3}.tcon.weights = [1 -1];
matlabbatch{3}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{4}.tcon.name = 'SPEM .4 > SPEM .2';
matlabbatch{3}.spm.stats.con.consess{4}.tcon.weights = [-1 1];
matlabbatch{3}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.delete = 1;


%% save individual batch for each participant
batch_name = ['first_level_' subject_name '.mat'];
save (batch_name, 'matlabbatch');


%% run batch
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, '');

end


