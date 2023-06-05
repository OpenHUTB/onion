function MakeFeatureSpace_NeuroSynth(ID)
%
%Obtain cognitive factor features
%
%Input:
% ID... subject ID (e.g., 'subject-01')
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Load CTM
load([SaveDir 'CTM_' ID '.mat'],'CTM');

% Load stimulus information (Matrix of Time points * features)
Method = 'TaskType';
load([DataDir 'Stim_' Method '_' ID '.mat']);
stimTrn = StimData.stimTrn;
stimVal = StimData.stimVal;

% Transform into the cognitive factor space
StimData.stimTrn = stimTrn*CTM;
StimData.stimVal = stimVal*CTM;

save([DataDir 'Stim_CogFactor_' ID '.mat'],'StimData', '-v7.3');