function Ridge_CerebCortex_RepFit(ID, Method)
%
%Determine the best regularization parameter
%by performing ridge regression repN (e.g., 50) times
%within the training dataset
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'TaskType' or 'CogFactor'
%

% Random seed
rng(1234,'twister')

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end

SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Load target voxels in the cerebral cortex
load([DataDir 'VsetInfo_' ID '.mat'],'vset_info');
ROI = vset_info.IDs;
vset = ROI(1);
voxelSetForm = [DataDir 'vset_' ID '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');
datasize = [96 96 72];

% Load response data (Matrix of Time points * Voxels)
load([DataDir 'RespData_' ID '.mat']);
respTrn_ROI = RespData.respTrn;

% Load stimulus information (Matrix of Time points * features)
load([DataDir 'Stim_' Method '_' ID '.mat']);
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

% Calculate best as from 18 candidate, for repN times
repN = 50;
SSinds = getResampInd_repN(size(respTrn_ROI,1), repN);
Abase = 100;
as = Abase*(2.^(0:17));
ttccs = zeros(1,length(as));
CCS = [];
for nn = 1:repN
    % Get random part
    ssinds = SSinds{nn};
    % Model fitting
    ws = ridgemulti(stimTrn(ssinds(1).trnInd,:), respTrn_ROI(ssinds(1).trnInd,:), as);
    % Validation step    
    ccs=zeros(length(tvoxels),length(as));
    for ii=1:length(as)
        presp = stimTrn(ssinds(1).regInd,:)*ws(:,:,ii);
        ccs(:,ii) = mvncorr(presp, respTrn_ROI(ssinds(1).regInd,:));
        fprintf('alpha=%12.2f, ccs = %.3f\n', as(ii), nanmean(ccs(:,ii)));
    end
    CCS{nn} = ccs;
    ttccs = ttccs + nanmean(ccs);  
end

% Average for repetitions
mccs = ttccs/repN;
% Select the best a from repetitions
[baccs, baind] = max(mccs);

rResult.Abase = Abase;
rResult.as = as;
rResult.repN = repN;
rResult.mccs = mccs;
rResult.CCS = CCS;
rResult.best_a = as(baind);

save([SaveDir 'RidgeRepFit_' Method '_' ID '.mat'], 'rResult', '-v7.3');
 

