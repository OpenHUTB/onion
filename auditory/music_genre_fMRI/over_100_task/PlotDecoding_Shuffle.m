function PlotDecoding_Shuffle(ID, Method)
%
%Plotting of decoding result using novel tasks,
%Shuflling CTM for 1000 times
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

% Load decoding result
load([SaveDir 'RidgeDecResult_' Method '_' ID '_NewTask_All.mat'], 'Result');
pstim = Result.pstim;

% Load cognitive transform matrix
load([SaveDir 'CTM_' ID '.mat'],'CTM');


% Randomize CTM
RepN = 1000;       
rng(1234,'twister')
rAcc = []; 
rTaskScore = [];
for rr = 1:RepN
    % Randomize task order
    if mod(rr,10) ==0
        display([' Resampling n = ' num2str(rr) ])
    end
    
    for ii = 1:103
        Rn = randperm(prod(size(CTM))); 
        rCTM = reshape(CTM,length(Rn),1);
        rCTM = rCTM(Rn);
        rCTM = reshape(rCTM, size(CTM));    
        rTaskVec = rCTM(ii,:);
        for tt = 1:size(pstim,1)
            rTaskScore(tt,ii) = corr2(pstim(tt,:),rTaskVec); 
        end   
    end

    for ii = 1:103
        LeftTask = setdiff(1:103,ii);
        TaskID = find(Label_vec==ii); 
        Score = [];
        for tt = LeftTask
            ScoreA = nanmean(rTaskScore(TaskID,ii),1);
            ScoreB = nanmean(rTaskScore(TaskID,tt),1);
            if ScoreA > ScoreB
                Score = [Score 1];
            else
                Score = [Score 0];
            end
        end
        rAcc(ii,rr) = nanmean(Score);  
    end
end
tccs = nanmean(rAcc); 


% Transform to ratio 
tccs_ratio = [];
bin = 0.01;
for pp = 0:bin:1
    Ntask = ceil(100*length( intersect(find(tccs>pp), find(tccs<=pp+bin))) / length(tccs));
    tccs_ratio = [tccs_ratio, (pp+0.0001)*ones(1,Ntask)];
end


dResult.rAcc_RnCTM = rAcc;
dResult.tccs = tccs;
dResult.RepN = RepN;
dresult.tccs_ratio = tccs_ratio;

fig = figure;
histogram(tccs_ratio,'BinWidth',0.01,'FaceColor','red')
xlabel('Decoding accuracy')
ylabel('Frequency (%)')
ylim([0,100]);
xlim([0.3,0.7]);
hold on
plot([0.5, 0.5],[0,100],'black')
set(gca,'XTick',[0.3, 0.5, 0.7])
set(gca,'YTick',[0, 100])

print([SaveDir 'DecodingResult_Shuffle_' ID '.eps' ],'-depsc')          
save([SaveDir 'DecodingResult_Shuffle_' ID '.mat'],'dResult')

