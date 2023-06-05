function PCAvisualize_RGB(w)
%
%Visualization of PCA result 
%based on the whole-cortex representation of all subjects
%
%Input:
% w... weight matrix obtained by ridge regression (concatenated for all subjects)
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Average for three time delays
nDelay = 3;
nFeature = size(w,1)/nDelay;
mw = reshape(w, nFeature,nDelay,[]);
mw = nanmean(mw,2);
mw = squeeze(mw);

% Normalize for each feature
smw = [];
for ff = 1:size(mw,1)
    smw(ff,:) = zscore(mw(ff,:));
end

% Apply PCA
[coeff, score, latent,tsquared,explained,mu] = pca(smw'); 

% Load task list
load([DataDir 'TaskList.mat'],'TaskList');

% Sort high PC weight tasks
pcaTask =[];
for N = 1:(nFeature-1)
    [sorted ind]= sort(coeff(:,N),'descend');
    pcaTask{N} = TaskList(ind);
end

pcaResult.coeff = coeff;
pcaResult.score = score;
pcaResult.latent = latent;
pcaResult.tsquared = tsquared;
pcaResult.explained = explained;

save([SaveDir 'GroupPCA_pcaTask'], 'pcaTask', '-v7.3');
save([SaveDir 'GroupPCA_Result'], 'pcaResult', '-v7.3')

%Specify target PC dimensions
PCAdim = [1, 2];

% Specify RGB color for each task based on PC1-PC3
Rdata = coeff(:,1);
Gdata = coeff(:,2);
Bdata = coeff(:,3);
Rdata = zscore(Rdata);
Rdata = Rdata + abs(min(Rdata));
Rdata = Rdata / max(Rdata);
Gdata = zscore(Gdata);
Gdata = Gdata + abs(min(Gdata));
Gdata = Gdata / max(Gdata);
Bdata = zscore(Bdata);
Bdata = Bdata + abs(min(Bdata));
Bdata = Bdata / max(Bdata);

% Visualize on 2D map with PC1 and PC2
PC1 = zscore(coeff(:,PCAdim(1)));
PC2 = zscore(coeff(:,PCAdim(2)));

ShowTask = 1:103;

% Scatter plot of tasks
Fig = figure('Position', [1, 1, 1200, 1000], 'InvertHardcopy', 'off');
set(0,'defaultAxesFontName','Helvetica'); 
Fig.Color = [0,0,0];
Fig.PaperPositionMode = 'auto';
MaxDotSize = 400;

fTaskW = ones(103,1);
scatter(PC1, PC2, MaxDotSize*fTaskW/3, [Rdata, Gdata, Bdata], 'filled'); hold on
axis off
xlim([min(PC1),max(PC1)])
ylim([min(PC2),max(PC2)])

FontSize = 15;
for tt = 1:nFeature
    if ismember(tt,ShowTask)
        text(PC1(tt),PC2(tt),TaskList(tt),'HorizontalAlignment','left','FontSize',FontSize,'color','white');
    end
end

print(Fig,[SaveDir 'PCAResult_Dots_pc' num2str(PCAdim(1)) '_pc' num2str(PCAdim(2))  ], '-depsc', '-r0');

