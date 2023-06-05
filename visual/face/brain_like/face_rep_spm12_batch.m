% This batch script analyses the Face fMRI dataset available from the SPM
% website:
%   http://www.fil.ion.ucl.ac.uk/spm/data/face_rep/
% as described in the SPM manual:
%   http://www.fil.ion.ucl.ac.uk/spm/doc/spm12_manual.pdf#Chap:data:faces
%__________________________________________________________________________
% Copyright (C) 2014-2015 Wellcome Trust Centre for Neuroimaging

% Guillaume Flandin
% $Id: face_rep_spm12_batch.m 17 2015-03-06 11:24:19Z guillaume $

% Directory containing Face data
%--------------------------------------------------------------------------
data_path = fullfile(pwd, 'face_rep');
if ~exist(data_path, 'dir')
    fprintf('%-40s:', 'Downloading Face dataset...');
    urlwrite('http://www.fil.ion.ucl.ac.uk/spm/download/data/face_rep/face_rep.zip','face_rep.zip');
    unzip(fullfile(data_path,'face_rep.zip'));
    fprintf(' %30s\n', '...done');
end
% if isempty(data_path), data_path = pwd; end


% Initialise SPM
%--------------------------------------------------------------------------
addpath(fullfile( matlabroot, 'utils', 'spm12'));
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
% spm_get_defaults('cmdline',true);

% Change working directory (useful for PostScript (.ps) files only)
%--------------------------------------------------------------------------
%clear matlabbatch
%matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_cd.dir = cellstr(data_path);
%spm_jobman('run',matlabbatch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPATIAL PREPROCESSING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear matlabbatch

% Select functional and structural scans
%--------------------------------------------------------------------------
f = spm_select('FPList', fullfile(data_path,'RawEPI'), '^sM.*\.img$');
a = spm_select('FPList', fullfile(data_path,'Structural'), '^sM.*\.img$');

% Realign
%--------------------------------------------------------------------------
matlabbatch{1}.spm.spatial.realign.estwrite.data{1} = cellstr(f);

% Slice Timing Correction
%--------------------------------------------------------------------------
matlabbatch{2}.spm.temporal.st.scans{1} = cellstr(spm_file(f,'prefix','r'));
matlabbatch{2}.spm.temporal.st.nslices = 24;
matlabbatch{2}.spm.temporal.st.tr = 2;
matlabbatch{2}.spm.temporal.st.ta = 2-2/24;
matlabbatch{2}.spm.temporal.st.so = 24:-1:1;
matlabbatch{2}.spm.temporal.st.refslice = 12;

% Coregister
%--------------------------------------------------------------------------
matlabbatch{3}.spm.spatial.coreg.estimate.ref    = cellstr(spm_file(f(1,:),'prefix','mean'));
matlabbatch{3}.spm.spatial.coreg.estimate.source = cellstr(a);

% Segment
%--------------------------------------------------------------------------
matlabbatch{4}.spm.spatial.preproc.channel.vols  = cellstr(a);
matlabbatch{4}.spm.spatial.preproc.channel.write = [0 1];
matlabbatch{4}.spm.spatial.preproc.warp.write    = [0 1];

% Normalise: Write
%--------------------------------------------------------------------------
matlabbatch{5}.spm.spatial.normalise.write.subj.def      = cellstr(spm_file(a,'prefix','y_','ext','nii'));
matlabbatch{5}.spm.spatial.normalise.write.subj.resample = cellstr(char(spm_file(f,'prefix','ar'),spm_file(f(1,:),'prefix','mean')));
matlabbatch{5}.spm.spatial.normalise.write.woptions.vox  = [3 3 3];

matlabbatch{6}.spm.spatial.normalise.write.subj.def      = cellstr(spm_file(a,'prefix','y_','ext','nii'));
matlabbatch{6}.spm.spatial.normalise.write.subj.resample = cellstr(spm_file(a,'prefix','m','ext','nii'));
matlabbatch{6}.spm.spatial.normalise.write.woptions.vox  = [1 1 1.5];

% Smooth
%--------------------------------------------------------------------------
matlabbatch{7}.spm.spatial.smooth.data = cellstr(spm_file(f,'prefix','war'));
matlabbatch{7}.spm.spatial.smooth.fwhm = [8 8 8];

%save('face_batch_preprocessing.mat','matlabbatch');
spm_jobman('run',matlabbatch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLASSICAL STATISTICAL ANALYSIS (CATEGORICAL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear matlabbatch

% Load onsets
%--------------------------------------------------------------------------
onsets    = load(fullfile(data_path,'sots.mat'));
condnames = {'N1' 'N2' 'F1' 'F2'};

% Output Directory
%--------------------------------------------------------------------------
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(data_path);
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'categorical';

% Model Specification
%--------------------------------------------------------------------------
matlabbatch{2}.spm.stats.fmri_spec.dir = cellstr(fullfile(data_path,'categorical'));
matlabbatch{2}.spm.stats.fmri_spec.timing.units = 'scans';
matlabbatch{2}.spm.stats.fmri_spec.timing.RT = 2;
matlabbatch{2}.spm.stats.fmri_spec.timing.fmri_t = 24;
matlabbatch{2}.spm.stats.fmri_spec.timing.fmri_t0 = 12;
matlabbatch{2}.spm.stats.fmri_spec.sess.scans = cellstr(spm_file(f,'prefix','swar'));
for i=1:numel(condnames)
    matlabbatch{2}.spm.stats.fmri_spec.sess.cond(i).name = condnames{i};
    matlabbatch{2}.spm.stats.fmri_spec.sess.cond(i).onset = onsets.sot{i};
    matlabbatch{2}.spm.stats.fmri_spec.sess.cond(i).duration = 0;
end
matlabbatch{2}.spm.stats.fmri_spec.sess.multi_reg   = cellstr(spm_file(f(1,:),'prefix','rp_','ext','.txt'));
matlabbatch{2}.spm.stats.fmri_spec.fact(1).name     = 'Fame';
matlabbatch{2}.spm.stats.fmri_spec.fact(1).levels   = 2;
matlabbatch{2}.spm.stats.fmri_spec.fact(2).name     = 'Rep';
matlabbatch{2}.spm.stats.fmri_spec.fact(2).levels   = 2;
matlabbatch{2}.spm.stats.fmri_spec.bases.hrf.derivs = [1 1];

% Model Estimation
%--------------------------------------------------------------------------
matlabbatch{3}.spm.stats.fmri_est.spmmat = cellstr(fullfile(data_path,'categorical','SPM.mat'));

save(fullfile(data_path,'categorical_spec.mat'),'matlabbatch');

% Inference
%--------------------------------------------------------------------------
matlabbatch{4}.spm.stats.results.spmmat = cellstr(fullfile(data_path,'categorical','SPM.mat'));
matlabbatch{4}.spm.stats.results.conspec.contrasts = Inf;
matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'FWE';

matlabbatch{5}.spm.stats.results.spmmat = cellstr(fullfile(data_path,'categorical','SPM.mat'));
matlabbatch{5}.spm.stats.results.conspec.contrasts  = 3;
matlabbatch{5}.spm.stats.results.conspec.threshdesc = 'none';
matlabbatch{5}.spm.stats.results.conspec.thresh     = 0.001;
matlabbatch{5}.spm.stats.results.conspec.extent     = 0;
matlabbatch{5}.spm.stats.results.conspec.mask.contrasts = 5;
matlabbatch{5}.spm.stats.results.conspec.mask.thresh    = 0.001;
matlabbatch{5}.spm.stats.results.conspec.mask.mtype     = 0;

matlabbatch{6}.spm.stats.con.spmmat = cellstr(fullfile(data_path,'categorical','SPM.mat'));
matlabbatch{6}.spm.stats.con.consess{1}.fcon.name = 'Movement-related effects';
matlabbatch{6}.spm.stats.con.consess{1}.fcon.weights = [zeros(6,3*4) eye(6)];
matlabbatch{7}.spm.stats.results.spmmat = cellstr(fullfile(data_path,'categorical','SPM.mat'));
matlabbatch{7}.spm.stats.results.conspec.contrasts = 17;
matlabbatch{7}.spm.stats.results.conspec.threshdesc = 'FWE';

% Run
%--------------------------------------------------------------------------
save('face_batch_categorical.mat','matlabbatch');
%spm_jobman('interactive',matlabbatch);
spm_jobman('run',matlabbatch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLASSICAL STATISTICAL ANALYSIS (PARAMETRIC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear matlabbatch

% Load onsets
%--------------------------------------------------------------------------
onsets = load(fullfile(data_path,'sots.mat'));

% Output directory
%--------------------------------------------------------------------------
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(data_path);
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'parametric';

% Model Specification (copy and edit the categorical one)
%--------------------------------------------------------------------------
batch_categ = load(fullfile(data_path,'categorical_spec.mat'));
matlabbatch{2} = batch_categ.matlabbatch{2};

matlabbatch{2}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name',{},'param',{},'poly',{});
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(2).pmod.name = 'Lag';
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(2).pmod.param = onsets.itemlag{2};
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(2).pmod.poly = 2;
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(3).pmod = struct('name',{},'param',{},'poly',{});
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(4).pmod.name = 'Lag';
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(4).pmod.param = onsets.itemlag{4};
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(4).pmod.poly = 2;
matlabbatch{2}.spm.stats.fmri_spec.dir = cellstr(fullfile(data_path,'parametric'));
matlabbatch{2}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];

% Model Estimation
%--------------------------------------------------------------------------
matlabbatch{3}.spm.stats.fmri_est.spmmat = cellstr(fullfile(data_path,'parametric','SPM.mat'));

% Inference
%--------------------------------------------------------------------------
matlabbatch{4}.spm.stats.con.spmmat = cellstr(fullfile(data_path,'parametric','SPM.mat'));
matlabbatch{4}.spm.stats.con.consess{1}.fcon.name = 'Famous Lag';
matlabbatch{4}.spm.stats.con.consess{1}.fcon.weights = [zeros(2,6) eye(2)];

matlabbatch{5}.spm.stats.results.spmmat = cellstr(fullfile(data_path,'parametric','SPM.mat'));
matlabbatch{5}.spm.stats.results.conspec.contrasts = Inf;
matlabbatch{5}.spm.stats.results.conspec.threshdesc = 'FWE';

matlabbatch{6}.spm.stats.results.spmmat = cellstr(fullfile(data_path,'parametric','SPM.mat'));
matlabbatch{6}.spm.stats.results.conspec.contrasts  = 9;
matlabbatch{6}.spm.stats.results.conspec.threshdesc = 'none';
matlabbatch{6}.spm.stats.results.conspec.thresh     = 0.001;
matlabbatch{6}.spm.stats.results.conspec.extent     = 0;
matlabbatch{6}.spm.stats.results.conspec.mask.contrasts = 5;
matlabbatch{6}.spm.stats.results.conspec.mask.thresh    = 0.05;
matlabbatch{6}.spm.stats.results.conspec.mask.mtype     = 0;

% Run
%--------------------------------------------------------------------------
save('face_batch_parametric.mat','matlabbatch');
%spm_jobman('interactive',matlabbatch);
spm_jobman('run',matlabbatch);
