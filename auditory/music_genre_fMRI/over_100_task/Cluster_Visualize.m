function Cluster_Visualize(w)
%
% Hierarchical clustering analysis and visualization
%
%Input:
% w... weight matrix obtained by ridge regression
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
resp = mw;

% Load task names
load([DataDir 'TaskList.mat'],'TaskList');

% Hierarchical clustering analysis with dissimilarity matrix
X = pdist(resp,'correlation');
Y = linkage(X, 'average');
fig = figure;
[T H outperm] = dendrogram(Y,0);
permtask = TaskList(outperm);
set(gca,'XTickLabel',permtask,'FontSize',10);
set(gca,'XTickLabelRotation',290);
set(gcf,'Position',[0, 0, 1800,800])
saveas(fig,[SaveDir 'Dendrogram.eps'],'epsc');


% Reorder tasks based on the clustering analysis
permresp = resp(outperm,:);
RespCorr = corr(permresp');

% Transform to 100 percentile
X = tril(RespCorr);
XX =X(find(X ~=0));
[XV XI] = sort(XX,'descend');
YY = zeros(size(XX));
rr = 1;
while rr < length(XV) + 1
    YY(XI(rr)) = rr;
    if rr < length(XV)
        c = 1;
        while XV(rr) == XV(rr+c)
            YY(XI(rr+c)) = rr;
            c = c + 1;
        end
        rr = rr + c - 1;
    end
    rr = rr + 1;
end

YY = length(YY) + 1 - YY;
XXrank = 100 * YY / length(YY);
trilInd = find(X ~= 0);
pRespCorr = zeros(size(RespCorr));
pRespCorr(trilInd) = XXrank;
uTril = tril(pRespCorr,-1);
pRespCorr = pRespCorr + uTril';


% Plot similarity matrix
Fig = figure('Color','white');
set(gcf,'Position',[0, 0, 1500,1500])
imagesc(pRespCorr)
colormap('parula')
cb = colorbar;
set(gca,'YTickLabel',permtask,'FontSize',10)
set(gca,'YTick',1:length(permtask))
set(gca,'XTickLabel',permtask,'FontSize',10)
set(gca,'XTick',1:length(permtask))
set(gca,'XTickLabelRotation',290);
caxis([0 100])
set(cb,'Ticks',[0,100])
set(gcf,'Position',[0, 0, 1500,1500])
axis square
saveas(Fig,[SaveDir 'SimilarityMatrix.eps'],'epsc');

% Save data
dResult.PermInd = outperm;
dResult.OrigTaskList = TaskList;
dResult.PermTaskList = permtask;
dResult.RespCorr = RespCorr;
dResult.pRespCorr = pRespCorr;
save([SaveDir 'ClusteringResult.mat'],'dResult')



