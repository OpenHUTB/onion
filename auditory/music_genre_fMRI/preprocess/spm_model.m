%% 广义线性模型 定义、估计、推断、结果

function spm_model(dir, runId)

cur_dataset = 'Training';
if runId > 12
    cur_dataset = 'Test';
end

% 第一个参数为`ExtList`表示从4D NIfTI文件中选择数据帧（只有文件名，ExtFPListRec返回完整路径），而`FPList`表示只选择文件
% '^swr.*\.nii$'
dir_splits = split(dir, filesep);
f = spm_select('ExtFPListRec', fullfile(dir, 'func'), sprintf('^swr%s_task-%s_run-%02d_bold.*\.nii$', char(dir_splits(length(dir_splits)-1)), cur_dataset, runId));
% if runId > 12
%     f = spm_select('ExtFPListRec', fullfile(dir, 'func'), sprintf('^swrsub-001_task-Test_run-%02d_bold.*\.nii$', runId));
% else
%     f = spm_select('ExtFPListRec', fullfile(dir, 'func'), sprintf('^swrsub-001_task-Training_run-%02d_bold.*\.nii$', runId));
% end

% Test_run-02(5)看不到、Test_run-03(6) 有一点点，没有04
% Training_run-01 几乎没有

% 不能所有Run都放进来一起进行广义线性回归，否则显示的是全脑到处都是激活。
% f = spm_select('ExtFPListRec', fullfile(pwd,'ds003720-download', 'sub-001', 'func'), '^swr.*\.nii$');

clear matlabbatch
% 
% Output Directory
cur_workspace_name = sprintf('classical_%s_run-%02d', cur_dataset, runId);
cur_workspace_dir = fullfile(dir, 'func', cur_workspace_name);
if exist(cur_workspace_dir, 'dir')
    delete(cur_workspace_dir);
end
mkdir(cur_workspace_dir);

matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(fullfile(dir, 'func'));
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = cur_workspace_name;  % 创建目录？

%% 1. 定义模型 Model Specification
matlabbatch{2}.spm.stats.fmri_spec.dir = cellstr(cur_workspace_dir);
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
spm_mat_path = fullfile(cur_workspace_dir,'SPM.mat');
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

% 
% % 使用xjview查看结果
% cd ..
xjview( fullfile(cur_workspace_dir, 'spmT_0001.nii') ); % /sub-01/spmT_0001.nii


pause(3);
close('all');  % 显示图形三秒后关闭
cd(fileparts(fileparts(fileparts(dir))))  % 当前工作路径回退到脚本所在的路径


end

