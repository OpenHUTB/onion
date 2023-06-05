function PC2Cluster
%
% Calculation of the relative contribution of the top PCs to the largest clusters
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];


Method = 'TaskType';
rng(1234,'twister')
RepN = 1000;

% Load label weight for each PC
load([ SaveDir 'GroupPCA_Result.mat']);
coeff = pcaResult.coeff;
TargPC = 1:4;
TargCluster = 1:6;
coeff = coeff(:,TargPC);

tCon = [];
pCon = [];
for Cluster =  TargCluster

    switch Cluster
        case 1
            TargetTask = 1:36; 
        case 2
            TargetTask = 37:49;
        case 3
            TargetTask = 50:64;
        case 4
            TargetTask = 65:77;
        case 5
            TargetTask = 78:86;
        case 6
            TargetTask = 87:103;
    end
    
    % Load task sets in the target cluster
    load([SaveDir 'ClusteringResult.mat'],'dResult');
    TargetTask = dResult.PermInd(TargetTask);

    % Relative contribution of top PCs to the target cluster
    Con = coeff(TargetTask,:);
    mCon = mean(Con,1);

    % Statistical testing (sign test)
    for PC = TargPC
        pCon(Cluster,PC) = signtest(Con(:,PC));
    end   
    tCon= [tCon;mCon];
end


Ps = reshape(pCon,prod(size(pCon)),1);
Ps = Ps';
%FDR correction
Q = 0.05; %default Q value
[PXsorted PXind] = sort(Ps, 'ascend');
NBox = length(TargPC)*length(TargCluster);
FDRthr = Q*[1:NBox]/NBox;
Diff = PXsorted - FDRthr;
thrInd = PXind(max(find(Diff<0)));
Thr = Ps(thrInd);
NS = find(Ps >= Thr);
nsCon = ones(size(pCon));
nsCon(NS) = 0;
tCon(NS) = 0;

WarmBase = [ 1, 0, 0];
CoolBase = [0, 0, 1];
labW = rgb2lab(WarmBase);
labC = rgb2lab(CoolBase);
Col1 = [(labW(1):(-labW(1)/49):0)', ones(50,1)*labW(2), ones(50,1)*labW(3)];
Col2 = [(0:labC(1)/49:labC(1))', ones(50,1)*labC(2), ones(50,1)*labC(3)];
rgbCol1 = lab2rgb(Col1);
rgbCol2 = lab2rgb(Col2);
rgbCol1(find(rgbCol1<0)) = 0;
rgbCol2(find(rgbCol2<0)) = 0;
map = [rgbCol1;[0.8,0.8,0.8];rgbCol2];
map = flipud(map);

Fig = figure('Color','white');
imagesc(tCon)
colormap(map)
cb = colorbar;
set(gca,'YTickLabel',{'Visual','Memory','Language','Motor','Introspection','Auditory'})
set(gca,'XTickLabel',{'PC1 (auditory)', 'PC2 (audio-visual)', 'PC3 (language)', 'PC4 (introspection)'})
set(gca,'XTickLabelRotation',-70)
cb.Ticks = [-0.2,0,0.2];
set(gca,'TickLength',[0,0])
caxis([-0.2,0.2])

cResult.tCon = tCon;
cResult.pCon = pCon;
save([SaveDir 'PC2Cluster_Group.mat'], 'cResult')
%saveas(Fig,[SaveDir 'PC2Cluster_Group.eps'],'epsc');
saveas(Fig,[SaveDir 'PC2Cluster_Group.png'],'png');

