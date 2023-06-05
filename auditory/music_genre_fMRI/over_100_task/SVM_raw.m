function SVM_raw(ID,Method)
%
% Decoding tasks using suport vector machine
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'TaskType'
%
%Requirement: LIBSVM v.3.22, https://github.com/cjlin1/libsvm 
%

addpath('./libsvm-3.22');
addpath('./libsvm-3.22/matlab')

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

TrnLabel = [];
for tt = 1:size(stimTrn,1)
    if ~isempty(find(stimTrn(tt,:)))
       TrnLabel(tt) = find(stimTrn(tt,:));
    else
       TrnLabel(tt) = 0;
    end
end
ValLabel = [];
for tt = 1:size(stimVal,1)
    if ~isempty(find(stimVal(tt,:)))
       ValLabel(tt) = find(stimVal(tt,:));
    else
       ValLabel(tt) = 0;
    end
end
ExcludeTrn = find(TrnLabel==0);
ExcludeVal = find(ValLabel==0);
TrnLabel(ExcludeTrn) = [];
respTrn_ROI(ExcludeTrn,:) = [];
ValLabel(ExcludeVal) = [];
respVal_ROI(ExcludeVal,:) = [];

tTrnLabel = [];
tTrnResp = [];
tt = 1;
tcount = 1;
while tt < length(TrnLabel)+1

    if tt == 1
        TmpLabel = TrnLabel(tt);
        TmpResp = respTrn_ROI(tt,:);
    elseif TrnLabel(tt) == TrnLabel(tt-1)
        TmpLabel = [TmpLabel, TrnLabel(tt)];
        TmpResp = [TmpResp; respTrn_ROI(tt,:)];
    else
        tTrnLabel(tcount) = nanmean(TmpLabel);
        tTrnResp(tcount,:) = nanmean(TmpResp,1);
        tcount = tcount + 1;
        TmpLabel = TrnLabel(tt);
        TmpResp = respTrn_ROI(tt,:);
    end
    tt = tt + 1;
end


tValLabel = [];
tValResp = [];
tt = 1;
tcount = 1;
while tt < length(ValLabel)+1
    if tt == 1
        TmpLabel = ValLabel(tt);
        TmpResp = respVal_ROI(tt,:);
    elseif ValLabel(tt) == ValLabel(tt-1)
        TmpLabel = [TmpLabel, ValLabel(tt)];
        TmpResp = [TmpResp; respVal_ROI(tt,:)];
    else
        tValLabel(tcount) = nanmean(TmpLabel);
        tValResp(tcount,:) = nanmean(TmpResp,1);
        tcount = tcount + 1;
        TmpLabel = ValLabel(tt);
        TmpResp = respVal_ROI(tt,:);
    end
    tt = tt + 1;
end


TrnLabel = tTrnLabel';
ValLabel = tValLabel';
TrnLabel = TrnLabel - 2;
ValLabel = ValLabel - 2;
TrnResp = double(tTrnResp);
ValResp = double(tValResp);


tAcc = nan(103);
% Pair-wise SVM
for ii = 1:103
    for jj = ii+1:103
        TaskIndA = find(TrnLabel==ii);
        TaskIndB = find(TrnLabel==jj);
        TrnLabelA = TrnLabel(TaskIndA);
        TrnLabelB = TrnLabel(TaskIndB);
        TrnRespA = TrnResp(TaskIndA,:);
        TrnRespB = TrnResp(TaskIndB,:);
        TrnLabelAB = [TrnLabelA; TrnLabelB];
        TrnRespAB = [TrnRespA; TrnRespB];

        TaskIndA = find(ValLabel==ii);
        TaskIndB = find(ValLabel==jj);
        ValLabelA = ValLabel(TaskIndA);
        ValLabelB = ValLabel(TaskIndB);
        ValRespA = ValResp(TaskIndA,:);
        ValRespB = ValResp(TaskIndB,:);
        ValLabelAB = [ValLabelA; ValLabelB];
        ValRespAB = [ValRespA; ValRespB];

        % Training SVM
        model = svmtrain( TrnLabelAB,TrnRespAB,  '-t 0');
        [PrdLabel, ~, probEstimates] = svmpredict(ValLabelAB, ValRespAB, model);
        tAcc(ii,jj) = sum((PrdLabel - ValLabelAB)==0)/length(PrdLabel);
    end
end
tAcc(isnan(tAcc)) = 0;
tAcc = tAcc + triu(tAcc)';

Acc= []; Ps = [];
for ii = 1:103
    Score = tAcc(ii,:);
    Score(ii ) = [];
    Ps(ii) = signtest(Score,0.5,'tail','right');
    Acc(ii) = nanmean(Score);
end


% Color only significantly decoded tasks
%FDR correction
Q = 0.05; %default Q value
[PXsorted PXind] = sort(Ps, 'ascend');
FDRthr = Q*[1:103]/103;
Diff = PXsorted - FDRthr;
thrInd = PXind(max(find(Diff<0)));
Thr = Ps(thrInd);

NS = find(Ps > Thr);

fig = figure;
histogram(Acc,'BinWidth',0.05,'FaceColor','blue')
hold on
histogram(Acc(NS),'BinWidth',0.05,'FaceColor','white')
xlabel('Decoding accuracy')
ylabel('Number of tasks')
ylim([0,105]);
xlim([0,1]);
hold on
plot([0.5, 0.5],[0,105],'red')
set(gca,'XTick',[0, 0.5, 1])
set(gca,'YTick',[0,20, 40, 60, 80, 100])
print([SaveDir '/DecodingResult_SVM_' ID '.eps' ],'-depsc')

dResult.Acc = Acc;
dResult.Ps = Ps;
dResult.NS = NS;

save([SaveDir '/DecodingResult_SVM_' ID '.mat'],'dResult');
