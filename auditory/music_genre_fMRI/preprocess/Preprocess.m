% Preprocess.m
%
% 对听“音乐流派”的fMRI数据集进行预处理，在所提供的目录中搜索anatomy和ses-movie目录。
% This funciton preprocesses data of the studyforrest data set. It searches in
% the provided directory for the anatomy and ses-movie/func directories.
% 1. 时间校正（Slice timing）：不同切片的时间差异的调整（切片定时）
%        参考slice、其他slice
% 2. 头动校正（Realignment）：不同切片运动的调整
%        不同scan之间体素对应关系遭到破坏、血液动力学响应被头动引起的信号淹没
%        刚体变换6个头动参数的估计：3个方向的平移(x,y,z;mm）、3个方向的旋转。
%        同一被试者不同采样时间点上的3D脑对齐。
% 3. 空间标准化（Normalisation）：将功能性数据打包进模板空间
%        不同人之间的大脑在形状、大小等方面存在明显差别。
%        在一个标准脑空间中，使不同被试者脑图像中的同一像素代表相同的解剖位置（将每个个体脑放入一个公共的标准空间）。
%            标准脑空间——Talairach坐标系：将大脑压扁或者伸展到“鞋盒”里，并对每个激活焦点抽取其3D坐标（x,y,z）。
%            先使用简单的线性变换（仿射变换）进行粗匹配，再使用复杂的非线性变换精匹配。
%        问题
%           计算复杂度高：高精度算法匹配一个脑需要几个消失
%           个体之间的脑并非是一一映射关系，不可能有完全精确的匹配
%        解决
%           对空间标准化后的脑图像进行适当的平滑
%           使用变形场信息
% 4. 空间平滑（Smoothing）：提高信噪比
%        使残差项更符合高斯分布假设
%        减少标准化后剩余的个体间差异
%        半高宽：指的是吸收谱带高度最大处高度为一半时谱带的全宽，常用来表示能量密度。

%
% input:
%	dir		- 要处理受试者的目录（如果为空则处理当前目录）directory of the subject to preprocess (if empty processes the current directory)
%   runId   - 要处理的视频段索引（从1到8）。the run index to process (varies from 1 to 8)

function Preprocess(dir, runId)
    
%     if (~isempty(dir) && ~strcmp(dir(end),'/'))
%         dir = [dir '/'];
%     end

    curDir = '';
    if (~strcmp(dir(1),'/'))
        curDir = pwd;
        curDir = [curDir '/'];
    end

    % 确定卷的个数为410
%     V = spm_vol(fullfile(dir, 'func', 'sub-001_task-Training_run-01_bold.nii'));
    numOfVolumes = 410;  % 和第一个扫描对齐 1-8 movie segment: 451, 441, 438, 488, 462, 439, 542, 338
    
    dir_parts = split(dir, filesep);
    cur_sub = cell2mat(dir_parts(length(dir_parts)-1));
    anatomyScan = fullfile(dir, 'anat', sprintf("%s_T1w.nii,1", cur_sub )); % [curDir dir 'anat/sub-001_T1w.nii,1'];  % 高分辨率解剖扫描文件

    scansDir = [dir 'func/'];
    name_list = strsplit(dir, filesep);
    subject = char(name_list(end-1));
    if runId > 12  % Test data
        regex = sprintf('^%s_task-Test_run-%02d_bold.nii', subject, runId-12);
    else
        regex = sprintf('^%s_task-Training_run-%02d_bold.nii', subject, runId);
    end
    
%     regex = [num2str(runId) '_bold.nii'];
    scans = spm_select('ExtFPList', scansDir, regex, 1:numOfVolumes);  % Return files matching the filter 'filt' and directories within 'direc'
    
    % remove 'meansub-01_ses-movie_task-movie_run-1_bold.nii' in ds000113/sub-01/ses-movie/func/
%     tmp_file = fullfile(dir, 'ses-movie', 'func', 'meansub-01_ses-movie_task-movie_run-1_bold.nii');
%     if exist(tmp_file)
%         delete(tmp_file);
%     end
    
    assert(size(scans,1) <= numOfVolumes, 'Detected more files than asked. Check if file is already preprocessed.');  % remove a '*1_bold.nii'
    if (size(scans,1) == 0)
        warning(['Could not find any scans for directory ' dir ' and run ID ' num2str(runId)]);
        return;
    end
    if runId > 12  % Test data
        raw_img_name = sprintf('%s_task-Test_run-%02d_bold.nii', subject, runId-12);
    else
        raw_img_name = sprintf('%s_task-Training_run-%02d_bold.nii', subject, runId);
    end
    raw_img = [curDir dir sprintf('func/%s,410', raw_img_name)];

    %% 1. 头动校正 Realign - Estimate and Write (mean)
    % In the coregistration step, the sessions are first realigned to each other, 
    % by aligning the first scan from each session to the first scan of the first 
    % session. The parameter estimation is performed this way because it is
    % assumed (rightly or not) that there may be systematic differences in
    % the images between sessions.
    % 在配准步骤中，首先进行重新对齐，将每个会话和第一个会话的第一个扫描进行对齐
    matlabbatch{1}.spm.spatial.realign.estwrite.data = {cellstr(scans)}; % Add new sessions for this subject
%     matlabbatch{1}.spm.spatial.realign.estwrite.data = {{'/data3/dong/brain/auditory/music_genre_fMRI/preprocess/ds003720-download/sub-001/func/sub-001_task-Training_run-01_bold.nii,410'}};
    % Options for Estimate and Reslice 一下都是默认值
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;  % 对质量和速度进行权衡。 quality versus speed trade-off
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;        % 采样点和参考图像之间间隔的距离（单位为毫米） the separation (in mm) between the points sampled in the reference image
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;       % the Full Width at Half Maximum of the Gaussian smoothing kernel (mm) applied to the images before estimating the realignment parameters.
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;     % The method (interpolation) by which the images are sampled when being written in a different space.
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0]; % directions in the volumes the values should wrap around in
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';    % Optional weighting image to weight each voxel of the reference image differently when estimating the realighment parameters
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];  % 仅仅是平均参数为[0 1]；which[2]: mean 1 -> no 'meansub-01_ses-movie_task-movie_run-1_bold.nii'
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;       % Because of subject motion, different images are likely to have different patterns of zeros from where it was not possible to sample data.
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';   % Specify the string to be prepended to the filenames of the resliced image file(s).

    %% 2. 对齐 Coregister - Estimate (Estimation Options)
    if runId > 12  % Test data
        ref_name = sprintf('mean%s_task-Test_run-%02d_bold.nii', subject, runId-12);
    else
        ref_name = sprintf('mean%s_task-Training_run-%02d_bold.nii', subject, runId);
    end
    ref_scan = [dir sprintf('func/%s,1', ref_name)];
    matlabbatch{2}.spm.spatial.coreg.estimate.ref = cellstr(ref_scan); % Reference Image: This is the image that is assumed to remain stationary (sometimes known as the target or template image), while the source image is moved to match it.
    matlabbatch{2}.spm.spatial.coreg.estimate.source = cellstr(anatomyScan);
    matlabbatch{2}.spm.spatial.coreg.estimate.other = {''};
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';  % Objective Function: Registration involves finding parameters that either maximise or minimise some objective function.
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];  % Separation: The average distance between sampled points (in mm). Can be a vector to allow a coarse registration followed by increasingly fine ones.
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];  % Tolerances: The accuracy for each parameter. Iterations stop when differences between successive estimates are less than the required tolerance.
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
    
    %% 3. 分割
    tpm_path = fullfile(matlabroot, 'software', 'matlab_utils', 'spm12', 'tpm', 'TPM.nii');
    matlabbatch{3}.spm.spatial.preproc.channel.vols = cellstr(anatomyScan);
    matlabbatch{3}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{3}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{3}.spm.spatial.preproc.channel.write = [0 1];
    matlabbatch{3}.spm.spatial.preproc.tissue(1).tpm = {[tpm_path ',1']};  % 需要/data2/whd/workspace/sot/hart/utils/spm12/tpm/TPM.nii
    matlabbatch{3}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{3}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(2).tpm = {[tpm_path ',2']};
    matlabbatch{3}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{3}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(3).tpm = {[tpm_path ',3']};
    matlabbatch{3}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{3}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(4).tpm = {[tpm_path ',4']};
    matlabbatch{3}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{3}.spm.spatial.preproc.tissue(4).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(5).tpm = {[tpm_path ',5']};
    matlabbatch{3}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{3}.spm.spatial.preproc.tissue(5).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(6).tpm = {[tpm_path ',6']};
    matlabbatch{3}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{3}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{3}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{3}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{3}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{3}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{3}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{3}.spm.spatial.preproc.warp.write = [0 1];
    matlabbatch{3}.spm.spatial.preproc.warp.vox = NaN;
    matlabbatch{3}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
        NaN NaN NaN];
    
    %%
    % 因为后面的空间正则化需要用到“再对齐”步骤中所生成的“r*.nii"文件
    spm_jobman('run', matlabbatch);
    clear matlabbatch
    
    
    %% 4. 空间正则化
    % 正则化功能像
    % Error reading header file "/data3/whd/data/neuro/music_genre/ds003720-download/sub-001/sub-001_task-Training_run-01_bold.nii".
    deformation_img = [dir sprintf('anat/y_%s_T1w.nii', subject)];
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = cellstr(deformation_img);
    if runId > 12  % Test data
        realign_regex = sprintf('^r%s_task-Test_run-%02d_bold.nii', subject, runId-12);
    else
        realign_regex = sprintf('^r%s_task-Training_run-%02d_bold.nii', subject, runId);
    end
    realign_scans = spm_select('ExtFPList', scansDir, realign_regex, 1:numOfVolumes);  % Return files matching the filter 'filt' and directories within 'direc'
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = cellstr(realign_scans);
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';
    
    % 正则化结构像
    matlabbatch{2}.spm.spatial.normalise.write.subj.def = cellstr(deformation_img);
    bias_corrected_img = [dir sprintf('anat/m%s_T1w.nii,1', subject)];
    matlabbatch{2}.spm.spatial.normalise.write.subj.resample = cellstr(bias_corrected_img);
    matlabbatch{2}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{2}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{2}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{2}.spm.spatial.normalise.write.woptions.prefix = 'w';
    
    %%
    % 因为后面的”平滑“需要用到“正则化”步骤中所生成的“wr*.nii"文件
    spm_jobman('run', matlabbatch);
    clear matlabbatch
    
    
    %% 5. 平滑
    if runId > 12  % Test data
        normalise_regex = sprintf('^wr%s_task-Test_run-%02d_bold.nii', subject, runId-12);
    else
        normalise_regex = sprintf('^wr%s_task-Training_run-%02d_bold.nii', subject, runId);
    end
    normalise_scans = spm_select('ExtFPList', scansDir, normalise_regex, 1:numOfVolumes);  % Return files matching the filter 'filt' and directories within 'direc'
%     smooth_img = [curDir dir sprintf('func/w%s_task-Training_run-%02d_bold.nii,410', subject, runId)];
    matlabbatch{1}.spm.spatial.smooth.data = cellstr(normalise_scans);
    matlabbatch{1}.spm.spatial.smooth.fwhm = [6 6 6];       % 高斯平滑核的半高宽（FWHM单位为毫米），表示xyz方向上的FWHM。 Full width at half maximum (FWHM) of the Gaussian smoothing kernel in mm. Three values should be entered, denoting the FWHM in the x, y and z directions.
    matlabbatch{1}.spm.spatial.smooth.dtype = 0;            % Data Type: Data type of the output images. 'SAME' indicates the same data type as the original images.
    matlabbatch{1}.spm.spatial.smooth.im = 0;               % Implicit masking: An 'implicit mask" is a mask implied by a particular voxel value (0 for images with integer type, NaN for float images).
    matlabbatch{1}.spm.spatial.smooth.prefix = 's';         % Filename prefix


    %% run created preprocess batch
    spm_jobman('run', matlabbatch);
end
