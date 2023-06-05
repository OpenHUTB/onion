function PCAvisualize_PosNeg(w, ID, voxelID)
%
%Visualization of task representation in each voxel,
%using PCA result
%
%Input:
% w... Group regression weight
% ID... subject ID (e.g., 'sub-01')
% voxelID.. Target voxel

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];
 
PCAdim = [1, 2];

% Load group PC result
load([SaveDir 'GroupPCA_Result.mat'], 'pcaResult');
load([SaveDir 'SubjectVoxels.mat'],'svoxels');

% Select voxels for target subject
Vstart = 1;
for ii = 1:6
    Vstart = [Vstart Vstart(ii) + length(svoxels{ii})];
end
w = w(:,Vstart(str2num(ID(6))):(Vstart(str2num(ID(6))+1)-1),:);

% Average for three time delays
nDelay = 3;
nFeature = size(w,1)/nDelay;
mw = reshape(w, nFeature,nDelay,[]);
mw = nanmean(mw,2);
mw = squeeze(mw);

% Weight of each task at the target voxel, keeping pos/neg
TaskW = mw(:,voxelID);
TaskW = TaskW / max(abs(TaskW));
TaskW_abs = abs(TaskW);
TaskW_pn = [];
WarmBase = [0.8, 0.2, 0.1];
CoolBase = [0.1, 0.2,0.8];
for tt = 1:length(TaskW)
    if TaskW(tt) >= 0
        TaskW_pn(tt,:) = WarmBase;
    else
        TaskW_pn(tt,:) = CoolBase;
   end
end

coeff = pcaResult.coeff(:,PCAdim);
PC1 = zscore(coeff(:,1));
PC2 = zscore(coeff(:,2) );

ShowTask = [45, 64,...
    46,34 72, 20, 27, 24, 75,...
    29, 93, 81,...
    54, 99, 84,...
    1, 6, 9,...
    13, 14,...
    21, 62, 88
];
% ShowTask = 1:103;

% Load label
load([DataDir 'TaskList.mat'],'TaskList');

Fig = figure('Position', [1, 1, 1200, 1000], 'InvertHardcopy', 'off');
set(0,'defaultAxesFontName','Helvetica'); 
Fig.Color = [0,0,0];
Fig.PaperPositionMode = 'auto';
MaxDotSize = 700;
scatter(PC1, PC2, MaxDotSize*TaskW_abs, TaskW_pn, 'filled'); hold on
axis off
xlim([min(PC1),max(PC1)])
ylim([min(PC2),max(PC2)])
FontSize = 15;

for tt = 1:nFeature
    if ismember(tt,ShowTask)
        text(PC1(tt),PC2(tt),TaskList(tt),'HorizontalAlignment','left','FontSize',FontSize,'color','white');
    end
end

print(Fig,[SaveDir 'PCAResult_PosNeg_' ID '_vID' num2str(voxelID) ], '-depsc', '-r0');
