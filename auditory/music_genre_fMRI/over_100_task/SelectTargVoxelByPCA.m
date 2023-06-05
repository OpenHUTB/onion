function voxelID = SelectTargVoxelByPCA(ID, TargPC)
%
% 从目标主成分中选择有代表性的体素
%
%Input:
%  ID... subject ID
%  TargPC... Target pricipal component where the top voxel will be selected

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];

% Load group PC result
load([SaveDir 'GroupPCA_Result.mat']);
load([SaveDir 'SubjectVoxels.mat'],'svoxels');

% Select voxels for target subject
Vstart = 1;
for ii = 1:6
    Vstart = [Vstart Vstart(ii) + length(svoxels{ii})];
end
% Get max score voxel of target PC
score = pcaResult.score(Vstart(str2num(ID(6))):(Vstart(str2num(ID(6))+1)-1),:);
[val voxelID] = max(score(:,TargPC));
