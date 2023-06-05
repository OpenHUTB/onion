function MDS_NonLinear
%
%Visualization of tasks on 2D space using nonlinear MDS
%
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];


MDSdim = [1, 2];

% Load task names
load([DataDir 'TaskList.mat'],'TaskList');

% Specify clusters
TaskID{1} = 1:35; 
TaskID{2} = 36:51;
TaskID{3} = 52:64;
TaskID{4} = 65:77;
TaskID{5} = 78:86;
TaskID{6} =  87:103;

load([SaveDir 'ClusteringResult_raw.mat'],'dResult')
Task =dResult.PermTaskList;

% Simirality matrix
SyMat = dResult.RespCorr;

% Dissimilarity matrix
D = 1 - SyMat; 
triD = tril(D);
triD = triD';
II = find(triD);
D(II) = triD(II);
Y = mdscale(D,2);

% Coordinates
Ax1 = double(Y(:,MDSdim(1)));
Ax2 = double(Y(:,MDSdim(2)));

Fig = figure('Position', [1, 1, 1200, 1000], 'Color','black','InvertHardcopy', 'off');

xlim([min(Ax1)-0.1,max(Ax1)+0.1])
ylim([min(Ax2)-0.1,max(Ax2)+0.1])
ColN = length(TaskID);

% Specify cluster colors
ColSet{1} = [1, 0.4, 0];
ColSet{2} = [1, 0.8, 0];
ColSet{3} = [0, 1, 0.4];   
ColSet{4} = [0, 1, 0.8];
ColSet{5} = [0, 0.4, 1];
ColSet{6} = [0, 0.8, 1];

MaxDotSize = 100;
scatter(Ax1(TaskID{1})+0.01,Ax2(TaskID{1}), MaxDotSize*ones(length(TaskID{1}),1), ColSet{1}, 'filled'); hold on
scatter(Ax1(TaskID{2})+0.01,Ax2(TaskID{2}), MaxDotSize*ones(length(TaskID{2}),1), ColSet{2}, 'filled'); 
scatter(Ax1(TaskID{3})+0.01,Ax2(TaskID{3}), MaxDotSize*ones(length(TaskID{3}),1), ColSet{3}, 'filled'); 
scatter(Ax1(TaskID{4})+0.01,Ax2(TaskID{4}), MaxDotSize*ones(length(TaskID{4}),1), ColSet{4}, 'filled'); 
scatter(Ax1(TaskID{5})+0.01,Ax2(TaskID{5}), MaxDotSize*ones(length(TaskID{5}),1), ColSet{5}, 'filled'); 
scatter(Ax1(TaskID{6})+0.01,Ax2(TaskID{6}), MaxDotSize*ones(length(TaskID{6}),1), ColSet{6}, 'filled'); 
axis off

FontSize = 15;
ShowTask = 1:103;
for tt = 1:103
    if ismember(tt,ShowTask)
        text(Ax1(tt)+0.01,Ax2(tt),Task(tt),'HorizontalAlignment','left','FontSize',FontSize,'color','white');
    end
end

print([SaveDir 'MDS_NL_dim' num2str(MDSdim(1)) '_dim' num2str(MDSdim(2)) '.eps'],'-depsc');




