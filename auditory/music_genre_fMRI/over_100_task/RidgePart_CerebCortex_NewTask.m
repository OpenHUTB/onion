function RidgePart_CerebCortex_NewTask(ID, Method, TargetTimeTrn,TargetTimeVal,Rep)
%
%Ridge regression analysis, using novel tasks
%within a subgroup
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'CogFactor'
% TargetTimeTrn... time points excluded from the training dataset
% TargetTimeVal... time points used in the test dataset 
% Rep... subgroup index

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
respVal_ROI = RespData.respVal;

% Load stimulus information (Matrix of Time points * features)
load([DataDir 'Stim_' Method '_' ID '.mat']);
stimTrn = StimData.stimTrn;
stimVal = StimData.stimVal;

% Normalization of Training Stimuli feature values
mu = nanmean(stimTrn);
sigma = nanstd(stimTrn);
stimTrn = bsxfun(@minus, stimTrn, mu);
stimTrn = bsxfun(@rdivide, stimTrn,sigma);

% Normalization of Validation Stimuli based on the Training norm value
stimVal = bsxfun(@minus, stimVal, mu);
stimVal = bsxfun(@rdivide, stimVal,sigma);

% NAN to zero
stimTrn(isnan(stimTrn))=0;
stimVal(isnan(stimVal))=0;

% Make delayed stimulus matrix
stimTrn = cat(2, circshift(stimTrn,1), circshift(stimTrn,2), circshift(stimTrn,3));
stimVal = cat(2, circshift(stimVal,1), circshift(stimVal,2), circshift(stimVal,3));

% Exclude time point around target tasks in training
stimTrn(TargetTimeTrn,:) = [];
respTrn_ROI(TargetTimeTrn,:) = [];

% Use time point around target tasks in test
stimVal = stimVal(TargetTimeVal,:);
respVal_ROI = respVal_ROI(TargetTimeVal,:);


% 10fold cross validation
ssinds = getResampInd10fold(size(respTrn_ROI,1));
Abase = 100;
as = Abase*(2.^(0:17));
tccs = zeros(1,18);
disp('model fitting...');
for dd = 1:10
    % Model fitting
    ws = ridgemulti(stimTrn(ssinds(dd).trnInd,:), respTrn_ROI(ssinds(dd).trnInd,:), as);
    % Validation step    
    ccs=zeros(length(tvoxels),length(as));
    for ii=1:length(as)
        presp = stimTrn(ssinds(dd).regInd,:)*ws(:,:,ii);
        ccs(:,ii) = mvncorr(presp, respTrn_ROI(ssinds(dd).regInd,:));
        fprintf('alpha=%12.2f, ccs = %.3f\n', as(ii), nanmean(ccs(:,ii)));
    end
    
    tccs = tccs + nanmean(ccs);
end
tccs = tccs/10;
% Select the best a from repetitions
[baccs, baind] = max(tccs);
fprintf('The best adlpha is %12.2f, ccs=%.3f.\n', as(baind), baccs);

% Calculate the final model and performance using test data
disp('calculating test performance...');
w = ridgemulti(stimTrn, respTrn_ROI, as(baind));
presp = stimVal*w;
ccs = mvncorr(presp, respVal_ROI); % test predictions
fprintf('mean ccs = %.3f\n', nanmean(ccs));


Result.ID = ID;
Result.Method = Method;
Result.datasize = datasize;    
Result.vset = vset;
Result.tvoxels = tvoxels;
Result.ROI = vset_info.labels(1);
Result.as = as;
Result.best_a = as(baind);
Result.w = w;
Result.ccs = ccs;
Result.mean_ccs = nanmean(ccs);
Result.resp = respVal_ROI;
Result.presp = presp;

save([SaveDir 'RidgeResult_' Method '_' ID '_NewTask' num2str(Rep) '.mat'], 'Result', '-v7.3');
  



    

