function Ridge_CerebCortex_NewTask_Dec_TopVoxels(ID, Method)
%
%Decoding analysis by ridge regression , using novel tasks
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'CogFactor'
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];

load([SaveDir 'NewTask_Target_Dec.mat'],'Targ');

TargetTimeTrn = Targ.TargetTimeTrn;
TargetTimeVal = Targ.TargetTimeVal;

% Select top voxels using training dataset 
Ntop = 1000;
load([SaveDir 'RidgeRepFit_' Method '_' ID '.mat'],'rResult');
[baval baind] = max(rResult.mccs);
tccs = [];
for rr = 1:50
    CCS = rResult.CCS{rr};
    tccs = [tccs,  CCS(:,baind)];
end
ccs = nanmean(tccs,2);
[val ind] = sort(ccs,'descend');
VoxelInd = ind(1:Ntop);
    
% Perform ridge regression for each of the five subgroups
for rr = 1:5
    RidgePart_CerebCortex_NewTask_Dec_TopVoxels(ID,Method, TargetTimeTrn{rr},TargetTimeVal{rr},rr,VoxelInd)
end
RidgePart_CerebCortex_NewTask_Dec_All_TopVoxels(ID, Method);
% Plot decoding result
PlotDecoding_TopVoxels(ID, Method)
