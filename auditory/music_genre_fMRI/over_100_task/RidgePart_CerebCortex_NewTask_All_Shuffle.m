function RidgePart_CerebCortex_NewTask_All_Shuffle(ID,Method)
%
%Shuffling ananalyis, it randomly shuffles feature matrix 1000 times in element-wise
%manner and produce a null distribution of mean prediction accuracy. Using
%novel tasks.
%
%Input: 
% ID... subject ID
% Method... 'CogFactor'

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Load target voxels in the cerebral cortex
load([DataDir 'VsetInfo_' ID '.mat'],'vset_info');
ROI = vset_info.IDs;
vset = ROI(1);
voxelSetForm = [DataDir 'vset_' ID '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');

% Load stimulus information (Matrix of Time points * features)
load([DataDir 'Stim_' Method '_' ID '.mat']);
stimTrn = StimData.stimTrn;
stimVal = StimData.stimVal;

% Normalization of Training Stimuli feature values
mu = nanmean(stimTrn);
sigma = nanstd(stimTrn);
% Normalization of Validation Stimuli based on the Training norm value
stimVal = bsxfun(@minus, stimVal, mu);
stimVal = bsxfun(@rdivide, stimVal,sigma);
% NAN to zero
stimVal(isnan(stimVal))=0;
% Make delayed stimulus matrix
stimVal = cat(2, circshift(stimVal,1), circshift(stimVal,2), circshift(stimVal,3));  

% Load target time points
load([SaveDir 'NewTask_Target.mat'],'Targ');
TargetTimeVal = Targ.TargetTimeVal;

% Concatenate time points of five subgroups
NewTime = [];
for rr = 1:5
    NewTime = [NewTime; TargetTimeVal{rr}];
end
[uniqueNT ia ic] = unique(NewTime);


AllResult= [];
for Rep = 1:5
    load([SaveDir  'RidgeResult_' Method '_' ID '_NewTask' num2str(Rep) '.mat'], 'Result');
    AllResult{Rep} = Result;
end


RepN = 1000;
rng(1234,'twister')
tccs = [];
for rr = 1:RepN
    tresp = [];
    tpresp = [];
    % Randomize task order
    if mod(rr,10) ==0
        display([' Resampling n = ' num2str(rr) ])
    end
    
    Rn = randperm(prod(size(stimVal)));  
    
    for Rep = 1:5
        % Use time point round target task in test dataset
        Rn_Targ = reshape(Rn, size(stimVal));
        Rn_Targ = Rn_Targ(TargetTimeVal{Rep},:);
        RnShape = size(Rn_Targ);
        Rn_Targ = reshape(Rn_Targ,prod(size(Rn_Targ)),1);
        rstimVal = stimVal(Rn_Targ);
        rstimVal = reshape(rstimVal, RnShape);  
        rpresp = rstimVal*AllResult{Rep}.w;
        tresp = [tresp; AllResult{Rep}.resp];
        tpresp = [tpresp; rpresp];
    end
    
    % Exclude duplicated time points
    ctpresp = [];
    ctresp = [];
    for tt = 1:length(uniqueNT)
        DoubledTime = find(NewTime == uniqueNT(tt));        
        ctpresp(min(NewTime(DoubledTime)),:) = nanmean(tpresp(DoubledTime,:),1);
        ctresp(min(NewTime(DoubledTime)),:) = nanmean(tresp(DoubledTime,:),1);        
    end  
    ccs = mvncorr(ctpresp, ctresp); 
    tccs(rr) = nanmean(ccs);
end


% Transform to ratio 
tccs_ratio = [];
bin = 0.005;
for pp = -1:bin:1
    Nvox = ceil(100*length( intersect(find(tccs>pp), find(tccs<=pp+bin))) / length(tccs));
    tccs_ratio = [tccs_ratio, (pp+0.0001)*ones(1,Nvox)];
end

pResult.RepN = RepN;
pResult.tccs = tccs;
pResult.tccs_ratio = tccs_ratio;

fig = figure;
histogram(tccs_ratio,'BinWidth',0.005,'FaceColor','red')
xlabel('Prediction accuracy')
ylabel('Frequency (%)')
ylim([0,100]);
xlim([-0.1,0.1]);
hold on
plot([0, 0],[0,100],'black')
set(gca,'XTick',[-0.1, 0, 0.1])
set(gca,'YTick',[0, 100])   
print([SaveDir 'RidgeResult_' Method '_' ID '_NewTask_All_Shuffle.eps' ],'-depsc')  

save([SaveDir 'RidgeResult_' Method '_' ID '_NewTask_All_Shuffle.mat' ], 'pResult', '-v7.3')

