function RidgePart_CerebCortex_NewTask_All(ID,Method)
%
%Concatenation of ridge regression analyses of subgroups 
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

% Load original response data
DummyMethod = 'TaskType';
load([SaveDir 'RidgeResult_' DummyMethod '_' ID '.mat'], 'Result');
oriresp = Result.resp;

% Load the regression results of five subgroups
tresp = [];
tpresp = [];
for Rep = 1:5
    load([SaveDir  'RidgeResult_' Method '_' ID '_NewTask' num2str(Rep) '.mat'], 'Result');
    tresp = [tresp; Result.resp];
    tpresp = [tpresp; Result.presp];
end

% Load target time points
load([SaveDir 'NewTask_Target.mat'],'Targ');
TargetTimeVal = Targ.TargetTimeVal;

% Concatenate time points of five subgroups
NewTime = [];
for rr = 1:5
    NewTime = [NewTime; TargetTimeVal{rr}];
end
[uniqueNT ia ic] = unique(NewTime);

% Concatenate predited responses of five subgroups
ctpresp = [];
for tt = 1:length(uniqueNT)
    DoubledTime = find(NewTime == uniqueNT(tt));
    ctpresp(min(NewTime(DoubledTime)),:) = nanmean(tpresp(DoubledTime,:),1);
end

% Calculate prediction performance
ccs = mvncorr(ctpresp, oriresp);

Result.ccs = ccs;  
Result.mean_ccs = nanmean(ccs);
Result.resp = oriresp;
Result.presp = ctpresp;

% Mapping from 1d Data to 3d .nii data 
RefEPI = [DataDir 'target_' ID '.nii'];
Y = zeros(prod(Result.datasize),1);
for ii=1:length(Result.tvoxels)
    Y(Result.tvoxels(ii))= ccs(ii);
end
vol = reshape(Y,Result.datasize);
vol_perm = permute(vol, [2,1,3]);
V = MRIread(RefEPI);
V.vol = vol_perm;
MRIwrite(V,[SaveDir 'RidgeResult_' Method '_' ID '_NewTask_All.nii']);


save([SaveDir 'RidgeResult_' Method '_' ID '_NewTask_All.mat'], 'Result', '-v7.3');
