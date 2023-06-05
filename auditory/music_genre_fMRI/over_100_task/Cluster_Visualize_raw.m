function Cluster_Visualize_raw
%
% Hierarchical clustering analysis and visualization, 
% using brain response data
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

gtResp = [];
for ii  = 1:6
    ID = ['sub-0' num2str(ii)];

    % Load response data (Matrix of Time points * Voxels)
    load([DataDir 'RespData_' ID '.mat']);
    respTrn_ROI = RespData.respTrn;

    % Correction for bold time delay
    RespTrn = cat(3,circshift(respTrn_ROI,-1), circshift(respTrn_ROI,-2), circshift(respTrn_ROI,-3));
    RespTrn = squeeze(mean(RespTrn,3));
    
    % Load stimulus information (Matrix of Time points * features)
    load([DataDir 'Stim_TaskType_' ID '.mat']);
    stimTrn = StimData.stimTrn;

    StimTrn = [];
    for ii = 1:size(stimTrn,1)
        if ~isempty(find(stimTrn(ii,:)))
            StimTrn(ii) = find(stimTrn(ii,:));
        else
            StimTrn(ii) = 0; 
        end
    end
    
    % Average for each task type
    tResp = [];
    NTask = 8;
    for tt = 1:103
        tdata = RespTrn( find(StimTrn==tt) , :);
        ttdata = reshape(tdata,size(tdata,1)/NTask,NTask,size(tdata,2));
        tResp = cat(1,tResp,nanmean(ttdata , 1) );
    end
    ttResp = tResp;
    tResp = squeeze( nanmean(ttResp,2) );
    gtResp = [gtResp,tResp];
end

% Load task names
load([DataDir 'TaskList.mat'],'TaskList');

% Hierarchical clustering analysis with dissimilarity matrix
X = pdist(gtResp,'correlation');
Y = linkage(X, 'average');
fig = figure;
[T H outperm] = dendrogram(Y,0);
permtask = TaskList(outperm);
set(gca,'XTickLabel',permtask,'FontSize',10);
set(gca,'XTickLabelRotation',290);
set(gcf,'Position',[0, 0, 1800,800])
saveas(fig,[SaveDir 'Dendrogram_raw.eps'],'epsc');


% Reorder tasks based on the clustering analysis
permresp = gtResp(outperm,:);
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
saveas(Fig,[SaveDir 'SimilarityMatrix_raw.eps'],'epsc');

% Save data
dResult.PermInd = outperm;
dResult.OrigTaskList = TaskList;
dResult.PermTaskList = permtask;
dResult.RespCorr = RespCorr;
dResult.pRespCorr = pRespCorr;
save([SaveDir 'ClusteringResult_raw.mat'],'dResult')



