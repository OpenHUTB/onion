% Preprocess.m
%
% 对“研究阿甘”数据集进行预处理，在所提供的目录中搜索anatomy和ses-movie目录。
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
    
    if (~isempty(dir) && ~strcmp(dir(end),'/'))
        dir = [dir '/'];
    end

    curDir = '';
    if (~strcmp(dir(1),'/'))
        curDir = pwd;
        curDir = [curDir '/'];
    end

    numOfVolumes = 451;  % 和第一个扫描对齐 1-8 movie segment: 451, 441, 438, 488, 462, 439, 542, 338
    
    anatomyScan = [curDir dir 'anatomy/highres001.nii,1'];  % 高分辨率解剖扫描文件

    scansDir = [curDir dir 'ses-movie/func/'];
    regex = [num2str(runId) '_bold.nii'];
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

    %% 2. 头动校正 Realign - Estimate and Write (mean)
    % In the coregistration step, the sessions are first realigned to each other, 
    % by aligning the first scan from each session to the first scan of the first 
    % session. The parameter estimation is performed this way because it is
    % assumed (rightly or not) that there may be systematic differences in
    % the images between sessions.
    % 在配准步骤中，首先进行重新对齐，将每个会话和第一个会话的第一个扫描进行对齐
    matlabbatch{1}.spm.spatial.realign.estwrite.data{1} = cellstr(scans); % Add new sessions for this subject
    % Options for Estimate and Reslice
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;  % 对质量和速度进行权衡。 quality versus speed trade-off
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;        % 采样点和参考图像之间间隔的距离（单位为毫米） the separation (in mm) between the points sampled in the reference image
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;       % the Full Width at Half Maximum of the Gaussian smoothing kernel (mm) applied to the images before estimating the realignment parameters.
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;     % The method (interpolation) by which the images are sampled when being written in a different space.
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0]; % directions in the volumes the values should wrap around in
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';    % Optional weighting image to weight each voxel of the reference image differently when estimating the realighment parameters
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [0 1];  % which[2]: mean 1 -> no 'meansub-01_ses-movie_task-movie_run-1_bold.nii'
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;       % Because of subject motion, different images are likely to have different patterns of zeros from where it was not possible to sample data.
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';   % Specify the string to be prepended to the filenames of the resliced image file(s).

    % 对齐 Coregister - Estimate (Estimation Options)
    matlabbatch{2}.spm.spatial.coreg.estimate.ref = cellstr(anatomyScan); % Reference Image: This is the image that is assumed to remain stationary (sometimes known as the target or template image), while the source image is moved to match it.
    matlabbatch{2}.spm.spatial.coreg.estimate.source(1) = cfg_dep('Realign: Estimate & Reslice: Mean Image', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rmean'));  % Source Image: This is the image that is assumed to remain stationary. 
    matlabbatch{2}.spm.spatial.coreg.estimate.other(1) = cfg_dep('Realign: Estimate & Reslice: Realigned Images (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','cfiles')); % Other Images: These are any images that need to remain in alilgnment with the source image.
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';  % Objective Function: Registration involves finding parameters that either maximise or minimise some objective function.
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];  % Separation: The average distance between sampled points (in mm). Can be a vector to allow a coarse registration followed by increasingly fine ones.
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];  % Tolerances: The accuracy for each parameter. Iterations stop when differences between successive estimates are less than the required tolerance.
    matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];

    %% 3. 空间正则化 Normalize - Estimate and write (Estimation Options and Writing Options)
    matlabbatch{3}.spm.spatial.normalise.estwrite.subj.vol = cellstr(anatomyScan);  % Subject: Data for this subject. The same parameters are used within subject.
    matlabbatch{3}.spm.spatial.normalise.estwrite.subj.resample(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
    matlabbatch{3}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;  % Bias regularisation: MR images are usually corrupted by a smooth, spatially varying artifact that modulates the intensity of the image (bias).
    matlabbatch{3}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;     % FWHM of Gaussian smoothness of bias.
    matlabbatch{3}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/data2/whd/workspace/sot/hart/utils/spm12/tpm/TPM.nii'};  % Tissue probability map: Select the tissue probability atlas.
    matlabbatch{3}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';    % Affine Regularisation: A Mutual Information affine registration with the tissue probability maps is used to achieve approximate alignment.
    matlabbatch{3}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];  % Warping Regularisation
    matlabbatch{3}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
    matlabbatch{3}.spm.spatial.normalise.estwrite.eoptions.samp = 3;  % Sampling distance: This encodes the approximate distance between sampled points when estimating the model parameters.
    matlabbatch{3}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
                                                                 78 76 85];  % Bounding box: The bounding box (in mm) of the volumn which is to be written (relative to the anterior commissure).
    matlabbatch{3}.spm.spatial.normalise.estwrite.woptions.vox = [3 3 3];    % Voxel sizes: The voxel sizes (x, y & z, in mm) of the written normalised images.
    matlabbatch{3}.spm.spatial.normalise.estwrite.woptions.interp = 4;       % Interpolation: The method by which the images are sampled when being written in a different space.
    matlabbatch{3}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';     % Filename Prefix: Specify the string to be prepended to the filenames of the normalised image file(s).
    matlabbatch{4}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Estimate & Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));  % Data: List of subjects. Images of each subject should be warped differently.

    %% 4. 空间平滑 Smooth data
    matlabbatch{4}.spm.spatial.smooth.fwhm = [8 8 8];  % 高斯平滑核的半高宽（FWHM单位为毫米），表示xyz方向上的FWHM。 Full width at half maximum (FWHM) of the Gaussian smoothing kernel in mm. Three values should be entered, denoting the FWHM in the x, y and z directions.
    matlabbatch{4}.spm.spatial.smooth.dtype = 0;       % Data Type: Data type of the output images. 'SAME' indicates the same data type as the original images.
    matlabbatch{4}.spm.spatial.smooth.im = 0;          % Implicit masking: An 'implicit mask" is a mask implied by a particular voxel value (0 for images with integer type, NaN for float images).
    matlabbatch{4}.spm.spatial.smooth.prefix = 's';    % Filename prefix

    %% run created matlab batch
    spm_jobman('run', matlabbatch);
end
