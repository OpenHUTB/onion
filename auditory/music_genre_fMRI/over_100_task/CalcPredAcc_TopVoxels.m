function CalcPredAcc_TopVoxels(ID, Method)
% Select top voxels from training dataset prediction accuracy 
% Novel task, using CogFacotor model

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end

SaveDir = [pwd '/SampleResult/'];

Ntop = 1000;
%Å@Select top voxels using traing dataset 
load([SaveDir 'RidgeRepFit_' Method '_' ID '.mat'], 'rResult');
[baval baind] = max(rResult.mccs);
tccs = [];
for rr = 1:50
    CCS = rResult.CCS{rr};
    tccs = [tccs,  CCS(:,baind)];
end
ccs = nanmean(tccs,2);

[val ind] = sort(ccs,'descend');

load([SaveDir 'RidgeResult_' Method '_' ID '_NewTask_All.mat'], 'Result'); 
mccs = mean(Result.ccs(ind(1:Ntop)));

display(['Mean accuracy of top ' num2str(Ntop) ' voxels is ' num2str(mccs) ])



