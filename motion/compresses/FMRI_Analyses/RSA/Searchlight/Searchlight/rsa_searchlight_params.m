function params = rsa_searchlight_params()

    params = struct();

    %% directories
    params.dir.projectDir = '/Volumes/Backup/fourrooms/';
    params.dir.inDir   = [params.dir.projectDir ''];
    params.dir.outDir  = [params.dir.projectDir 'analysis/Multivariate/RSA/Searchlight/CSQ_ModelFree_NDR/GOAL/Betas'];
    params.dir.maskDir = [params.dir.projectDir 'analysis/Templates/'];

    params.dir.subDir.SPM = '';%'dmats/';
    params.dir.subDir.RDM = 'rsa/';
    params.dir.subDir.GRP = 'grouplevel/';
    params.dir.subDir.rsabetas = 'betas/';

    %% file names
    params.names.subjectDir = '/Volumes/Backup/fourrooms/preprocessing/smooth_mni_6mm/sub-301/Models/CurrentRoom_byGoalState_byContext';%
    params.names.groupMask  = 'rGroupMask';%'groupMask_cval_searchlight';
    params.names.rdmSetIn   = 'brainRDMs_cval_searchlight';
    params.names.rdmSetOut  = 'brainRDMs_cval_searchlight';
    params.names.betasOut   = 'betas_cval_searchlight';
    params.names.models     = 'modelRDMs_cval_searchlight';
    params.names.modelset   = ''; % 

    %% model correlations
    params.corrs.modelset = str2num(params.names.modelset);
    params.corrs.method = 'spearman'; % kendall, regression (instead of correlations), spearmann
    params.corrs.whichruns = 'cval'; % 'avg', 'cval'.cval is full runxrun matrix, avg is the average of this over runs

    %% numbers
    params.num.subjects = 1; % number of subjects (o rly)
    params.num.runs = 2; % number of runs
    params.num.conditions = 23; % number of conditions
    params.num.motionregs = 10; % number of nuisance regressors
    params.num.badsSubjects = [0];
    params.num.runIDs = [8, 12, 16, 20, 24, 28];

    %% rsa
    params.rsa.method = 'searchlight'; % 'roi', 'searchlight'
    params.rsa.whichruns = 'cval'; % 'avg', 'cval'. avg is mean RDM across runs. If crossval is selected, creates nRunsxnRuns RDM (brain and models), where within run dissims are NaN
    params.rsa.metric = 'correlation'; % distance metric
    params.rsa.whiten = 1; % whiten betas (irrespective of dist measure)
    params.rsa.radius = 4; % radius of searchlight sphere (in voxels)

    %% hpc
    params.hpc.parallelise = 0;
    params.hpc.numWorkers = 2;

    %% statistical inference
    params.statinf.doFisher = 0; % fisher transformation (if method==spearman and test == t-test)
    params.statinf.threshVal = .05;
    params.statinf.threshStr = '005';
    params.statinf.method = 'ttest'; % signrank or t-test
    params.statinf.tail = 'right'; % right or both makes sense for modelcorrelations
