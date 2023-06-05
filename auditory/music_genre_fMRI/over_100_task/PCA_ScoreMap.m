function PCA_ScoreMap(ID, PC)
%
%Obtain cortical map of each PC
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% PC...Target principal component number
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Load selected voxels above threshold
load([SaveDir 'SubjectVoxels.mat'],'svoxels');

% Load group PCA result
load([ SaveDir 'GroupPCA_Result.mat']);
score = pcaResult.score;

% Load target voxels in the cerebral cortex
load([DataDir 'VsetInfo_' ID '.mat'],'vset_info');
ROI = vset_info.IDs;
vset = ROI(1);
voxelSetForm = [DataDir 'vset_' ID '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');
datasize = [96 96 72];
  
% Write down voxel values of the target PC into .nii
PCAdata_ori = zeros(size(tvoxels));
Nvoxel = length(svoxels{str2num(ID(6))});
PCAdata = score(1:Nvoxel,PC);
for kk = 1:length(PCAdata)
    PCAdata_ori(svoxels{str2num(ID(6))}(kk)) = PCAdata(kk);
end 

PCAdata_ori = zscore(PCAdata_ori);

% Transform 1d Data into 3d .nii file 
RefEPI = [DataDir 'target_' ID '.nii'];
Y = zeros(prod(datasize),1);
for vv=1:length(tvoxels)
   Y(tvoxels(vv))= PCAdata_ori(vv);
end
vol = reshape(Y,datasize);
vol_perm = permute(vol, [2,1,3]);
V = MRIread(RefEPI);
V.vol = vol_perm;
MRIwrite(V,[SaveDir 'GroupPCA_' ID '_PC' num2str(PC) '.nii']);
