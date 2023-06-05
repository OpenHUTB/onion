%% 广义线性模型 定义、估计、推断、结果
% 使用一个数据（sub-001/func/sub-001_task-Test_run-01_bold）进行测试和分析，
% 而main.m为自动化处理所有受试的数据。

run('../../../init_env.m')

spm('Defaults','fMRI');
spm_jobman('initcfg');

%%
% 第一个参数为`ExtList`表示从4D NIfTI文件中选择数据帧（只有文件名，ExtFPListRec返回完整路径），而`FPList`表示只选择文件
% '^swr.*\.nii$'
f = spm_select('ExtFPListRec', fullfile(pwd,'ds003720-download', 'sub-001', 'func'), '^swrsub-001_task-Test_run-01_bold.*\.nii$');
% Test_run-02(5)看不到、Test_run-03(6) 有一点点，没有04
% Training_run-01 几乎没有

% 不能所有Run都放进来一起进行广义线性回归，否则显示的是全脑到处都是激活。
% f = spm_select('ExtFPListRec', fullfile(pwd,'ds003720-download', 'sub-001', 'func'), '^swr.*\.nii$');

clear matlabbatch
% 
% Output Directory
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(pwd);
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'classical';  % 创建目录？

%% 1. 定义模型 Model Specification
matlabbatch{2}.spm.stats.fmri_spec.dir = cellstr(fullfile(pwd, 'classical'));
matlabbatch{2}.spm.stats.fmri_spec.timing.units = 'scans';
matlabbatch{2}.spm.stats.fmri_spec.timing.RT = 1.5;
matlabbatch{2}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{2}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

matlabbatch{2}.spm.stats.fmri_spec.sess.scans = cellstr(f);

matlabbatch{2}.spm.stats.fmri_spec.sess.cond.name = 'listening';
matlabbatch{2}.spm.stats.fmri_spec.sess.cond.onset = 0:10:400;
matlabbatch{2}.spm.stats.fmri_spec.sess.cond.duration = 10;

matlabbatch{2}.spm.stats.fmri_spec.sess.cond.tmod = 0;
matlabbatch{2}.spm.stats.fmri_spec.sess.cond.pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{2}.spm.stats.fmri_spec.sess.cond.orth = 1;
matlabbatch{2}.spm.stats.fmri_spec.sess.multi = {''};
matlabbatch{2}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
matlabbatch{2}.spm.stats.fmri_spec.sess.multi_reg = {''};
matlabbatch{2}.spm.stats.fmri_spec.sess.hpf = 128;
matlabbatch{2}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{2}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{2}.spm.stats.fmri_spec.volt = 1;
matlabbatch{2}.spm.stats.fmri_spec.global = 'None';
matlabbatch{2}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{2}.spm.stats.fmri_spec.mask = {''};
matlabbatch{2}.spm.stats.fmri_spec.cvi = 'AR(1)';


%% 2. 训练模型 Model Estimation
spm_mat_path = fullfile(pwd, 'classical','SPM.mat');
if exist(spm_mat_path, 'file')  % 如果之前存在了SPM.mat就删除，避免删除提示框出现
    delete(spm_mat_path);
end
matlabbatch{3}.spm.stats.fmri_est.spmmat = cellstr(spm_mat_path);
matlabbatch{3}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{3}.spm.stats.fmri_est.method.Classical = 1;
 
%% 3. 定义对比 Contrasts
matlabbatch{4}.spm.stats.con.spmmat = cellstr(spm_mat_path);
matlabbatch{4}.spm.stats.con.consess{1}.tcon.name = 'Listening > Rest';
matlabbatch{4}.spm.stats.con.consess{1}.tcon.weights = [1 0];
matlabbatch{4}.spm.stats.con.consess{2}.tcon.name = 'Rest > Listening';
matlabbatch{4}.spm.stats.con.consess{2}.tcon.weights = [-1 0];

%% 4. 推断结果 Inference Results
matlabbatch{5}.spm.stats.results.spmmat = cellstr(spm_mat_path);
matlabbatch{5}.spm.stats.results.conspec.contrasts = 1;
matlabbatch{5}.spm.stats.results.conspec.threshdesc = 'FWE';
matlabbatch{5}.spm.stats.results.conspec.thresh = 0.05;
matlabbatch{5}.spm.stats.results.conspec.extent = 0;
matlabbatch{5}.spm.stats.results.print = false;
 
%% 5. 显示结果 Rendering
matlabbatch{6}.spm.util.render.display.rendfile = {fullfile(spm('Dir'),'canonical','cortex_20484.surf.gii')};
matlabbatch{6}.spm.util.render.display.conspec.spmmat = cellstr(spm_mat_path);
matlabbatch{6}.spm.util.render.display.conspec.contrasts = 1;
matlabbatch{6}.spm.util.render.display.conspec.threshdesc = 'FWE';
matlabbatch{6}.spm.util.render.display.conspec.thresh = 0.05;
matlabbatch{6}.spm.util.render.display.conspec.extent = 0;

spm_jobman('run', matlabbatch);


%% 使用xjview查看结果
cd ..
xjview( fullfile(pwd, 'classical', 'spmT_0001.nii') ); % /sub-01/spmT_0001.nii

% overlay选择"Temporal_Sup_R_aal"表示只查看右脑伤颞的激活

