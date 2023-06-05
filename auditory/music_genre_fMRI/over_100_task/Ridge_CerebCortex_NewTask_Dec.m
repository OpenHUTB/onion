function Ridge_CerebCortex_NewTask_Dec(ID, Method)
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

% Perform ridge regression for each of the five subgroups
for rr = 1:5
    RidgePart_CerebCortex_NewTask_Dec(ID,Method, TargetTimeTrn{rr},TargetTimeVal{rr},rr)
end
RidgePart_CerebCortex_NewTask_Dec_All(ID, Method);
% Plot decoding result
PlotDecoding(ID, Method)
