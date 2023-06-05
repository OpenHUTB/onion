%
%Randomly assign tasks into five sub-groups
%Obtain target time points exclude from the training dataset
%Obtain target time points used in the test dataset
%


if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

rng(1234,'twister');
RN = randperm(103);
TargetTask{1} = RN(1:20);
TargetTask{2} = RN(21:40);
TargetTask{3} = RN(41:61);
TargetTask{4} = RN(62:82);
TargetTask{5} = RN(83:103);

RemainTask{1} = RN(21:103);
RemainTask{2} = [RN(1:20) RN(41:103)];
RemainTask{3} = [RN(1:40) RN(62:103)];
RemainTask{4} = [RN(1:61) RN(83:103)];
RemainTask{5} = RN(1:82);

% To get time period to be excluded in Training stim, and that used in the target
DummyID = 'sub-01';
DummyMethod = 'TaskType';
% Load target voxels in the cerebral cortex
load([DataDir 'VsetInfo_' DummyID '.mat'],'vset_info');
ROI = vset_info.IDs;
vset = ROI(1);
voxelSetForm = [DataDir 'vset_' DummyID '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');

% Load stimulus information (Matrix of Time points * features)
load([DataDir 'Stim_' DummyMethod '_' DummyID '.mat']);
stimTrn = StimData.stimTrn;
stimVal = StimData.stimVal;

TargetTimeTrn = [];
TargetTimeVal = [];
for rr = 1:5
    % Get all time points of TargetTask + 3 time delays in training dataset
    ttargTime = [];
    for tt = 1:length(TargetTask{rr})
        targ = TargetTask{rr}(tt);
        targTime = find(stimTrn(:,targ));
        targTime = [targTime;max(targTime)+[1; 2; 3]];
        ttargTime = [ttargTime; targTime];
    end
    ttargTime = unique(ttargTime);
    % Max Training Length = training clip size
    ttargTime(find(ttargTime>size(stimTrn,1))) = [];    
    TargetTimeTrn{rr} = ttargTime;
    
    % Get all time points of TargetTask + 3 time delays in the test dataset
    ttargTime = [];
    for tt = 1:length(TargetTask{rr})
        targ = TargetTask{rr}(tt);
        targTime = find(stimVal(:,targ));
        targTime = [targTime;max(targTime)+[1; 2; 3]];
        ttargTime = [ttargTime; targTime];
    end
    ttargTime = unique(ttargTime);
    % Max Validation Length = validation clip size
    ttargTime(find(ttargTime>size(stimVal,1))) = [];
    
    TargetTimeVal{rr} = ttargTime;    
end

Targ.TargetTask = TargetTask;
Targ.RemainTask = RemainTask;
Targ.TargetTimeTrn = TargetTimeTrn;
Targ.TargetTimeVal = TargetTimeVal;

save([SaveDir 'NewTask_Target.mat'],'Targ');


% Get all time points of TargetTask, without time delays (for decoding)
TargetTimeTrn = [];
TargetTimeVal = [];
for rr = 1:5
    ttargTime = [];
    for tt = 1:length(TargetTask{rr})
        targ = TargetTask{rr}(tt);
        targTime = find(stimTrn(:,targ));
        %targTime = [targTime;max(targTime)+[1; 2; 3]];
        ttargTime = [ttargTime; targTime];
    end
    ttargTime = unique(ttargTime);
    % Max Training Length = training clip size
    ttargTime(find(ttargTime>size(stimTrn,1))) = [];    
    TargetTimeTrn{rr} = ttargTime;
    
    ttargTime = [];
    for tt = 1:length(TargetTask{rr})
        targ = TargetTask{rr}(tt);
        targTime = find(stimVal(:,targ));
        %targTime = [targTime;max(targTime)+[1; 2; 3]];
        ttargTime = [ttargTime; targTime];
    end
    ttargTime = unique(ttargTime);
    % Max Validation Length = validation clip size
    ttargTime(find(ttargTime>size(stimVal,1))) = [];
    
    TargetTimeVal{rr} = ttargTime;    
end

Targ.TargetTimeTrn = TargetTimeTrn;
Targ.TargetTimeVal = TargetTimeVal;

save([SaveDir 'NewTask_Target_Dec.mat'],'Targ');
