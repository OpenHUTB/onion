function Ridge_FDRcorr(ID, Method)
%
%FDR correction for multiple comparison (Benjamini & Hochberg, 1995)
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% Method... 'TaskType'
%

% Random seed
rng(1234,'twister')

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
datasize = [96 96 72];

load([SaveDir 'RidgeResult_' Method '_' ID '.mat'], 'Result');
ccs = Result.ccs;

% Default Q value
Q = 0.05; 
% Make random correlation coefficient histogram
for ii = 1:length(ccs)
	A = normrnd(0,1,size(Result.resp,1),1);
	B = normrnd(0,1,size(Result.resp,1),1);
	rccs(ii) = corr2(A,B);
end

% Calculate P values in each voxel
PX = [];
for ii = 1:length(ccs)
	x = find(rccs>ccs(ii));
	px = length(x)/length(ccs);
	PX(ii) = px;
end

% FDR correction
[PXsorted PXind] = sort(PX, 'ascend');
FDRthr = Q*[1:length(ccs)]/length(ccs);
Diff = PXsorted - FDRthr;
thrInd = PXind(max(find(Diff<0)));
thrP = PX(thrInd);
thrCCs = ccs(thrInd);
disp(['Threshold ccs = ' num2str(thrCCs)])

% New ccs above threshold
ccs(find(ccs < thrCCs)) = 0;


% Mapping from 1d Data to 3d .nii data 
RefEPI = [DataDir 'target_' ID '.nii'];
Y = zeros(prod(datasize),1);
for ii=1:length(Result.tvoxels)
    Y(Result.tvoxels(ii))= ccs(ii);
end
vol = reshape(Y,datasize);
vol_perm = permute(vol, [2,1,3]);
V = MRIread(RefEPI);
V.vol = vol_perm;
MRIwrite(V,[SaveDir 'RidgeResult_' Method '_FDRcorr_' ID '.nii']);

Result.ccs_fdr = ccs;
Result.fdrthr = thrCCs;

save([SaveDir 'RidgeResult_' Method '_' ID '.mat'], 'Result', '-v7.3');
    

    

