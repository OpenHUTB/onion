function RidgePart_CerebCortex_NewTask_Dec(ID, Method, TargetTimeTrn,TargetTimeVal,Rep)
%
%Decoding analysis using ridge regression, 
%using novel tasks, within a subgroup
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

mstimVal = stimVal;
mstimTrn = stimTrn;

% Make stimulus with delays
DelayN = 3;
trespTrn_ROI = []; trespVal_ROI = [];
for tt = 1:DelayN
    trespTrn_ROI = cat(2, trespTrn_ROI, circshift(respTrn_ROI,-tt));
    trespVal_ROI = cat(2, trespVal_ROI, circshift(respVal_ROI,-tt));
end
respTrn_ROI = trespTrn_ROI;
respVal_ROI = trespVal_ROI;
        
% Exclude time point around target tasks in training
stimTrn(TargetTimeTrn,:) = [];
respTrn_ROI(TargetTimeTrn,:) = [];
mstimTrn(TargetTimeTrn,:) = [];

% Use time point around target tasks in validation
respVal_ROI = respVal_ROI(TargetTimeVal,:);
mstimVal = mstimVal(TargetTimeVal,:);


% 10fold cross validation
ssinds = getResampInd10fold(size(respTrn_ROI,1));
Abase = 100;
as = Abase*(2.^(0:17));
tccs = zeros(1,18);
for dd = 1:10
    % Model fitting
    ws = ridgemulti(respTrn_ROI(ssinds(dd).trnInd,:), stimTrn(ssinds(dd).trnInd,:), as);
    % Validation step     
    ccs=zeros(size(stimTrn,2),length(as));
    for ii=1:length(as)
        pstim = respTrn_ROI(ssinds(dd).regInd,:)*ws(:,:,ii);
        ccs(:,ii) = mvncorr(pstim, mstimTrn(ssinds(dd).regInd,:));
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
w = ridgemulti(respTrn_ROI, stimTrn, as(baind));
pstim = respVal_ROI*w;
ccs = mvncorr(pstim, mstimVal);
ccs_tmp = mvncorr(pstim', mstimVal');
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
Result.ccs_tmp = ccs_tmp;    
Result.mean_ccs = nanmean(ccs); 
Result.mstim = mstimVal;
Result.pstim = pstim;

save([SaveDir 'RidgeDecResult_' Method '_' ID '_NewTask' num2str(Rep)], 'Result');



    

