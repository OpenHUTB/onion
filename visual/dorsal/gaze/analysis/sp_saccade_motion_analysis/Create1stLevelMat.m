% Create1stLevelMat.m
%
% This function sets up the .mat file and executes it for the GLM analysis of
% the 1st level
% 建立.mat文件并执行个体水平的GLM分析。
% （对每个扫描文件进行广义线性模型分析）
% 
% 个体水平分析（对应 群体水平分析）
%     基本过程：实验设计->个体扫描->个体激活区检测
%     目的：对这个被试者，你感兴趣的效应（平稳跟踪）在哪些脑区出现，其强度如何？
% GLM
%     GLM的数学表达：观测数据=设计矩阵×参数+残差（Y = X × w + b）
%     设计矩阵X：行为刺激，列包括刺激因素G、干扰因素H（线性趋势、全局激活：比如头部运动参数）
%         刺激序列 × HRF = 设计矩阵中的刺激因素X
%         为什么要考虑这些干扰因素？线性趋势、可能的呼吸制品、头部运动参数
%
% input:
%   outputDir   - 保存一层次矩阵和估计结果的目录. directory to save 1st level matrix as well as result from estimation
%   scansDir    - 保存扫描的目录(func). directory that holds scans 
%   intsDir     - 回归变量所在目录(regressors). directory where intervals are stored
%   subjId      - 受试者ID. ID of the subject 

function Create1stLevelMat(outputDir, scansDir, intsDir, subjId)
    % 该实验中fMRI扫描仪的常数. constant of fMRI scanner for this experiment
    % 重复时间. repetition time
    rt = 2; % included in .json file
    numOfVolumes = 542;  % 为什么用最长的volume？ movie segments 1-8: 451, 441, 438, 488, 462, 439, 542, 338

    c_runIds = [1 2 3 4 5 6 7 8];
    c_spExt = '_sp.txt'; % 保存回归变量目录(regressors)中平稳跟踪的后缀
    c_saccExt = '_sacc.txt';
    c_motionExt = '_motion.txt';

    % assure that output dir exists
    if (~exist(outputDir))
        mkdir(outputDir)
    end

    % make sure subject is string
    assert(ischar(subjId), 'Subject id should be string');
    assert(length(subjId)==2, 'Subject id should have length 2');

    % 创建工作结构. create jobs structure
    matlabbatch{1,1}.spm.stats.fmri_spec.dir = cellstr(outputDir);

    matlabbatch{1,1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1,1}.spm.stats.fmri_spec.timing.RT = rt;
    matlabbatch{1,1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1,1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

    % 对每个子视频片段进行配置。  start of multisession part
    for sessId=1:length(c_runIds)
        runId = c_runIds(sessId);

        % 得到当前会话（视频片段）的扫描. get scans for the session
        filter = ['sws*run-' num2str(runId) '*'];  % 血氧浓度依赖文件(sws*)
        boldFile = glob([scansDir '/' filter]);
        assert(size(boldFile,1)==1, 'More than one or no files were found. Regexp shoulf be refined');
        [dir, basename, ext] = fileparts(boldFile{1});
        files = spm_select('ExtFPList', scansDir, basename, 1:numOfVolumes);
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).scans = cellstr(files);
        nscans = size(files,1);

        %% 根据回归变量(regressors)创建平稳跟踪条件结构. create sp condition struct
        spTimingFile = glob([intsDir '/sub-' subjId '*run-' num2str(runId) '*' c_spExt]);
        assert(size(spTimingFile,1)==1, 'Exactly one SP timing file should be found');
        sp = importdata(spTimingFile{1}, ',');

        spOnsets = sp(:,1)./1000000; % 平稳跟踪开始时间，原来的单位为微秒，转换为秒. convert to seconds
        spDurations = zeros(size(spOnsets)); % 原来持续时间为2秒，将其改为0，使其看起来像脉冲相应. make it like impulse response
        spValues = sp(:,3);  % 平稳跟踪的值.  (质量的置信度估计，1表示好的眼跟踪，0表示跟踪丢失xxx)
        spMod = struct('name', {'sp_modulation'}, 'param', {spValues}, 'poly', {1});  % 平稳跟踪调制参数（值）
        structSP = struct('name', {'SP'}, 'onset', {spOnsets}, 'duration', {spDurations}, 'tmod', {0}, 'pmod', {spMod}, 'orth', {1});

        %% 创建眼跳条件定时。 create saccade condition timing
        saccTimingFile = glob([intsDir '/sub-' subjId '*run-' num2str(runId) '*' c_saccExt]);
        assert(size(saccTimingFile,1)==1, 'Exactly one saccade timing file should be found');
        sacc = importdata(saccTimingFile{1}, ',');

        saccOnsets = sacc(:,1)./1000000; % convert to seconds
        saccDurations = zeros(size(saccOnsets)); % convert to impulse response
        saccValues = sacc(:,3);
        saccMod = struct('name', {'saccade_modulation'}, 'param', {saccValues}, 'poly', {1});
        structSacc = struct('name', {'saccade'}, 'onset', {saccOnsets}, 'duration', {saccDurations}, 'tmod', {0}, 'pmod', {saccMod}, 'orth', {1});

        %% 创建运动条件定时。 create motion condition timing
        motionTimingFile = glob([intsDir '/sub-' subjId '*run-' num2str(runId) '*' c_motionExt]);
        assert(size(motionTimingFile,1)==1, 'Exactly one motion timing file should be found');
        motion = importdata(motionTimingFile{1}, ',');

        motionOnsets = motion(:,1)./1000000; %convert to seconds
        motionDurations = zeros(size(motionOnsets)); % convert to impulse response
        motionValues = motion(:,3);
        motionMod = struct('name', {'motion_modulation'}, 'param', {motionValues}, 'poly', {1});
        structMotion = struct('name', {'motion'}, 'onset', {motionOnsets}, 'duration', {motionDurations}, 'tmod', {0}, 'pmod', {motionMod}, 'orth', {1});

        %% 输入前面构造的三个条件
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).cond = [structSP, structMotion, structSacc];  % condition
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).multi = {''};
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).regress = struct([]);
        alignFile = glob([scansDir '/rp*run-' num2str(runId) '*.txt']);
        assert(size(alignFile,1)==1, 'Too many or too few rp_.. files');
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).multi_reg = cellstr(alignFile);  % 多个回归变量. multiple regressors
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).hpf = 128;  % 高通滤波器. High Pass Filter (HPF). e default HPF cut-off of 128s or 0.008Hz

    end
    % end of multisession part

    matlabbatch{1,1}.spm.stats.fmri_spec.fact = struct([]);

    matlabbatch{1,1}.spm.stats.fmri_spec.bases.hrf = struct('derivs', {[1,1]}); % 血液动力学相应函数(HRF). 时间和分散梯度. time and dispersion derivatives
    
    matlabbatch{1,1}.spm.stats.fmri_spec.volt = 1;

    matlabbatch{1,1}.spm.stats.fmri_spec.global = 'None';
    
    matlabbatch{1,1}.spm.stats.fmri_spec.mthresh = 0.8;

    matlabbatch{1,1}.spm.stats.fmri_spec.mask = {''};

    matlabbatch{1,1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    save([ outputDir '/1st_level_file.mat'], 'matlabbatch');  % 保存批处理的配置信息

    % 为估计配置matlab批处理（保存成SPM.mat文件）. configure matlab batch for estimation (saves SPM.mat)
    spm_jobman('run', matlabbatch);  % 高层接口

    % 估计广义线性模型（为条件创建beta.nii，为残差创建ResMS.nii）. estimate GLM (creates beta.nii for conditions and ResMS.nii for residual)
    initialPath = pwd;
    cd(outputDir); % spm_spm works if we are at the same directory as SPM.mat
    load SPM;
    spm_spm(SPM);  % 广义线性模型的估计

    % move to initial path
    cd(initialPath);


end
