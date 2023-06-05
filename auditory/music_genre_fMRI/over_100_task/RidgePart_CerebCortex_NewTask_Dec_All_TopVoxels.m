function RidgePart_CerebCortex_NewTask_Dec_All_TopVoxels(ID,Method)
%
%Concatenation of decoding analyses using ridge regression, 
%using novel tasks
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'CogFactor'

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Load the regression results of five subgroups
tmstim= [];
tpstim = [];
for Rep = 1:5
    load([SaveDir  'RidgeDecResult_TopVoxels_' Method '_' ID '_NewTask' num2str(Rep) '.mat'], 'Result');
    tmstim = [tmstim; Result.mstim];
    tpstim = [tpstim; Result.pstim];
end

% Load target time points
load([SaveDir 'NewTask_Target_Dec.mat'],'Targ');
TargetTimeVal = Targ.TargetTimeVal;

% Concatenate time points of five subgroups
NewTime = [];
for rr = 1:5
    NewTime = [NewTime; TargetTimeVal{rr}];
end
[uniqueNT ia ic] = unique(NewTime);

% Concatenate decoded features of five subgroups
pstim = [];
mstim = [];
for tt = 1:length(uniqueNT)
    DoubledTime = find(NewTime == uniqueNT(tt));
    pstim(tt,:) = nanmean(tpstim(DoubledTime,:),1);
    mstim(tt,:) = nanmean(tmstim(DoubledTime,:),1);   
end

ccs = mvncorr(pstim, mstim);
ccs_tmp = mvncorr(pstim', mstim');

Result.ccs = ccs;
Result.mean_ccs = nanmean(ccs);
Result.ccs_tmp = ccs_tmp;
Result.mstim = mstim;
Result.pstim = pstim;


save([SaveDir 'RidgeDecResult_TopVoxels_' Method '_' ID '_NewTask_All.mat'], 'Result', '-v7.3');

