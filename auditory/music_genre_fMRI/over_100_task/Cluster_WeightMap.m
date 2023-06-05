function Cluster_WeightMap(ID, Cluster)
%
%Obtrain cortical map of average weight values of target hierarhical
%cluster
%
%Input:
% ID... subject ID
% Cluster... target cluster index

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];


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

% Load target voxels in the cerebral cortex
load([DataDir 'VsetInfo_' ID '.mat'],'vset_info');
ROI = vset_info.IDs;
vset = ROI(1);
voxelSetForm = [DataDir 'vset_' ID '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');
datasize = [96 96 72];

% Load weight data of target subject
Method = 'TaskType';
load([SaveDir 'RidgeResult_' Method '_' ID '.mat'], 'Result');
w = Result.w;

% Average three time delays of all tasks in the target cluster
mw = reshape(w,size(w,1)/3,3,size(w,2));
mw = squeeze(nanmean(mw,2));
mw = mw(TargetTask,:);
mw = nanmean(mw,1);
mw = zscore(mw);

%Transform 1d Data into 3d .nii file 
RefEPI = [DataDir 'target_' ID '.nii'];
Y = zeros(prod(datasize),1);
for ii=1:length(tvoxels)
    Y(tvoxels(ii))= mw(ii);
end
vol = reshape(Y,datasize);
vol_perm = permute(vol, [2,1,3]);
V = MRIread(RefEPI);
V.vol = vol_perm;
MRIwrite(V,[SaveDir 'GroupCluster_' ID '_Cluster' num2str(Cluster) '.nii']);




