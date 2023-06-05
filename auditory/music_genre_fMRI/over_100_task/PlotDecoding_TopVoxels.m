function PlotDecoding_TopVoxels(ID, Method)
%
%Plotting of decoding result using novel tasks
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'CogFactor'
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Load stimulus information (Matrix of Time points * features)
DummyMethod = 'TaskType';
load([DataDir 'Stim_' DummyMethod '_' ID '.mat']);
stimTrn = StimData.stimTrn;
stimVal = StimData.stimVal;

% Get label index at each time bin
mstim = [];
for tt = 1:size(stimVal,1)
    if ~isempty(find(stimVal(tt,:)))
        mstim(tt) = find(stimVal(tt,:));
    else
        mstim(tt) = 0; 
    end
end

% Load decoding result
load([SaveDir 'RidgeDecResult_TopVoxels_' Method '_' ID '_NewTask_All.mat'], 'Result');
pstim = Result.pstim;

% Load cognitive transform matrix
load([SaveDir 'CTM_' ID '.mat'],'CTM');

% Calculate task-score 
TaskScore = [];
for ii = 1:103
   TaskVec = CTM(ii,:);
   for tt = 1:size(pstim,1)
      TaskScore(tt,ii) = corr2(pstim(tt,:),TaskVec); 
   end
end        

% Exclude no task time point (feedback time point)
Label_vec = mstim;
for tt = 1:length(Label_vec)
    if tt > length(Label_vec)
        break;   
    elseif Label_vec(tt) == 0
        Label_vec(tt) = [];
        tt = tt -1;
    end
end

% Decoding accuracy for each trial (Acc)
Acc = [];
Ps = [];
for ii = 1:103
    LeftTask = setdiff(1:103,ii);
    TaskID = find(Label_vec==ii); 
    Score = [];
    for tt = LeftTask
        ScoreA = nanmean(TaskScore(TaskID,ii),1);
        ScoreB = nanmean(TaskScore(TaskID,tt),1);
        if ScoreA > ScoreB
            Score = [Score 1];
        else
            Score = [Score 0];
        end
    end
    Ps(ii) = signtest(Score,0.5,'tail','right');
    Acc(ii) = nanmean(Score);
end

% FDR correction
Q = 0.05;
[PXsorted PXind] = sort(Ps, 'ascend');
FDRthr = Q*[1:103]/103;
Diff = PXsorted - FDRthr;
thrInd = PXind(max(find(Diff<0)));
Thr = Ps(thrInd);
NS = find(Ps > Thr);

% Task-wise accuracy histogram, color only significantly decoded tasks
fig = figure;
histogram(Acc,'BinWidth',0.05,'FaceColor','blue')
hold on
histogram(Acc(NS),'BinWidth',0.05,'FaceColor','white')
xlabel('Decoding accuracy')
ylabel('Number of tasks')
ylim([0,100]);
xlim([0,1]);
hold on
plot([0.5, 0.5],[0,100],'red')
set(gca,'XTick',[0, 0.5, 1])
set(gca,'YTick',[0,20, 40, 60, 80, 100])
print([SaveDir 'DecodingResult_TopVoxels_' ID '.eps' ],'-depsc')   

dResult.Acc = Acc;
dResult.mAcc = nanmean(Acc);
dResult.NonSig = NS;
dResult.Ps = Ps;
dResult.FDRthr = Thr;
save([SaveDir 'DecodingResult_TopVoxels_' ID '.mat'], 'dResult')


      