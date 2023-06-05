% Create1stLevelMat.m
%
% This function sets up the .mat file and executes it for the GLM analysis of
% the 1st level
%
% input:
%   outputDir   - directory to save 1st level matrix as well as results from estimation
%   scansDir    - directory that holds scans 
%   intsDir     - directory where regressor intervals are stored
%   subjId      - ID of the subject 

function Create1stLevelMat(outputDir, scansDir, intsDir, subjId)
    % constant of fMRI scanner for this experiment
    % repetition time
    rt = 2; % included in .json file
    numOfVolumes = 542;

    c_runIds = [1 2 3 4 5 6 7 8];
    c_spExt = '_sp.txt';
    c_saccExt = '_sacc.txt';

    % assure that output dir exists
    if (~exist(outputDir))
        mkdir(outputDir)
    end

    % make sure subject is string
    assert(ischar(subjId), 'Subject id should be string');
    assert(length(subjId)==2, 'Subject id should have length 2');

    % create jobs structure
    matlabbatch{1,1}.spm.stats.fmri_spec.dir = cellstr(outputDir);

    matlabbatch{1,1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1,1}.spm.stats.fmri_spec.timing.RT = rt;
    matlabbatch{1,1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1,1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

    % start of multisession part
    for sessId=1:length(c_runIds)
        runId = c_runIds(sessId);

        % get scans for the session
        filter = ['sws*run-' num2str(runId) '*'];
        boldFile = glob([scansDir '/' filter]);
        assert(size(boldFile,1)==1, 'More than one or no files were found. Regexp shoulf be refined');
        [dir, basename, ext] = fileparts(boldFile{1});
        files = spm_select('ExtFPList', scansDir, basename, 1:numOfVolumes);
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).scans = cellstr(files);
        nscans = size(files,1);

        % create sp condition struct
        spTimingFile = glob([intsDir '/sub-' subjId '*run-' num2str(runId) '*' c_spExt]);
        assert(size(spTimingFile,1)==1, 'Exactly one SP timing file should be found');
        sp = importdata(spTimingFile{1}, ',');

        spOnsets = sp(:,1)./1000000; % convert to seconds
        spDurations = zeros(size(spOnsets)); % make it like impulse response
        spValues = sp(:,3);
        spMod = struct('name', {'sp_modulation'}, 'param', {spValues}, 'poly', {1});
        structSP = struct('name', {'SP'}, 'onset', {spOnsets}, 'duration', {spDurations}, 'tmod', {0}, 'pmod', {spMod}, 'orth', {1});

        % create saccade condition timing
        saccTimingFile = glob([intsDir '/sub-' subjId '*run-' num2str(runId) '*' c_saccExt]);
        assert(size(saccTimingFile,1)==1, 'Exactly one saccade timing file should be found');
        sacc = importdata(saccTimingFile{1}, ',');

        saccOnsets = sacc(:,1)./1000000; %convert to seconds
        saccDurations = zeros(size(saccOnsets)); % convert to impulse response
        saccValues = sacc(:,3);
        saccMod = struct('name', {'saccade_modulation'}, 'param', {saccValues}, 'poly', {1});
        structSacc = struct('name', {'saccade'}, 'onset', {saccOnsets}, 'duration', {saccDurations}, 'tmod', {0}, 'pmod', {saccMod}, 'orth', {1});

        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).cond = [structSP, structSacc];
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).multi = {''};
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).regress = struct([]);
        alignFile = glob([scansDir '/rp*run-' num2str(runId) '*.txt']);
        assert(size(alignFile,1)==1, 'Too many or too few rp_.. files');
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).multi_reg = cellstr(alignFile);
        matlabbatch{1,1}.spm.stats.fmri_spec.sess(1,sessId).hpf = 128;

    end
    % end of multisession part

    matlabbatch{1,1}.spm.stats.fmri_spec.fact = struct([]);

    matlabbatch{1,1}.spm.stats.fmri_spec.bases.hrf = struct('derivs', {[1,1]}); % time and dispersion derivatives
    
    matlabbatch{1,1}.spm.stats.fmri_spec.volt = 1;

    matlabbatch{1,1}.spm.stats.fmri_spec.global = 'None';
    
    matlabbatch{1,1}.spm.stats.fmri_spec.mthresh = 0.8;

    matlabbatch{1,1}.spm.stats.fmri_spec.mask = {''};

    matlabbatch{1,1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    save([ outputDir '/1st_level_file.mat'], 'matlabbatch');

    % configure matlab batch for estimation (saves SPM.mat)
    spm_jobman('run', matlabbatch);

    % estimate GLM (creates beta.nii for conditions and ResMS.nii for residual)
    initialPath = pwd;
    cd(outputDir); % spm_spm works if we are at the same directory as SPM.mat
    load SPM;
    spm_spm(SPM);

    % move to initial path
    cd(initialPath);
end
