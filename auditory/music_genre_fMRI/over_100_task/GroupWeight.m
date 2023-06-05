function GroupWeight( Method )
%
%Select the best regularization parameter used in the principal component
%analysis and hierarchical clustering analysis
%
%Input:
% Method... 'TaskType'
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

DummyID = 'sub-01';
% Load target voxels in the cerebral cortex
load([DataDir 'VsetInfo_' DummyID '.mat'],'vset_info');
ROI = vset_info.IDs;
vset = ROI(1);
voxelSetForm = [DataDir 'vset_' DummyID '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');

% Load stimulus information (Matrix of Time points * features)
load([DataDir 'Stim_' Method '_' DummyID '.mat']);
stimTrn = StimData.stimTrn;

% Normalization of Training Stimuli feature values
mu = nanmean(stimTrn);
sigma = nanstd(stimTrn);
stimTrn = bsxfun(@minus, stimTrn, mu);
stimTrn = bsxfun(@rdivide, stimTrn,sigma);

% NAN to zero
stimTrn(isnan(stimTrn))=0;

% Make delayed stimulus matrix
stimTrn = cat(2, circshift(stimTrn,1), circshift(stimTrn,2), circshift(stimTrn,3));

% Load result of repetitive ridge to search for the best parameter
repN = 50;
tmccs = zeros(1,18);
for ii = 1:6
    ID = ['sub-0' num2str(ii)];    
    load([SaveDir 'RidgeRepFit_' Method '_' ID '.mat'], 'rResult');
    tmccs = tmccs + rResult.mccs;
end
tmccs = tmccs / 6;
%@Select the best regularization parameter for 6 participants
[baccs, baind] = max(tmccs);
best_a = rResult.as(baind);


tresp = [];
svoxels = [];
for ii = 1:6
    ID = ['sub-0' num2str(ii)];    
    load([SaveDir 'RidgeRepFit_' Method '_' ID '.mat'], 'rResult');
    tccs = zeros(size(rResult.CCS{1}(:,1)));
    for nn = 1:repN
        CCS = rResult.CCS{nn};
        tccs = tccs + CCS(:,baind);
    end
    
    % Load target voxels in the cerebral cortex
    load([DataDir 'VsetInfo_' ID '.mat'],'vset_info');
    ROI = vset_info.IDs;
    vset = ROI(1);
    voxelSetForm = [DataDir 'vset_' ID '/vset_%03d.mat'];
    load(sprintf(voxelSetForm,vset),'tvoxels');

    % Load significant voxels    
    load([SaveDir 'RidgeResult_' Method '_' ID '.mat'], 'Result');
    thr = Result.fdrthr;
    svoxels{ii} = find(Result.ccs>thr);
  
    % Use only selected tvoxels
    tvoxels = tvoxels(svoxels{ii});   

    % Load response data (Matrix of Time points * Voxels)
    load([DataDir 'RespData_' ID '.mat']);
    respTrn_ROI = RespData.respTrn;
    respTrn_ROI = respTrn_ROI(:,svoxels{ii});

    % Concatenate selected voxel responses
    tresp = [tresp, respTrn_ROI];

end


disp('calculating validation performance...');
w = ridgemulti(stimTrn, tresp, best_a);

save([SaveDir 'GroupRidgeWeight.mat'],'w', '-v7.3')
save([SaveDir 'SubjectVoxels.mat'],'svoxels', '-v7.3')



