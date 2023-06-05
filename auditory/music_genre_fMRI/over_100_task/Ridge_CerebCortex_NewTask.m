function Ridge_CerebCortex_NewTask(ID, Method)
%
%Ridge regression analysis, using novel tasks
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'CogFactor'
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];

load([SaveDir 'NewTask_Target.mat'],'Targ');

TargetTimeTrn = Targ.TargetTimeTrn;
TargetTimeVal = Targ.TargetTimeVal;

% Perform ridge regression for each of the five subgroups
for rr = 1:5
    RidgePart_CerebCortex_NewTask(ID,Method, TargetTimeTrn{rr},TargetTimeVal{rr},rr)
end

RidgePart_CerebCortex_NewTask_All(ID, Method);
Ridge_FDRcorr_NewTask(ID, Method);




