function PCA_NeuroSynth(ID, PC, NeuroSynthDir)
%
%Reverse infernce of cognitive factors related to each PC 
%
%Input:
% ID... subject ID (e.g., 'sub-01')
% PC...Target principal component number
% NeuroSynthDir...Directory where reference images in NeuroSynth were saved
%

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Load input PCA result
InputFile = [SaveDir 'GroupPCA_' ID '_PC' num2str(PC) '.nii'];
V = MRIread(InputFile);
vol = V.vol;
vol = permute(vol,[2,1,3]);
tdata =  reshape(vol,prod(size(vol)),1);

% Load registration matrix from  MNI152 space to EPI space
load([DataDir 'reg_mni152_' ID '.mat'],'R');

% Load neurosynth terms
load([DataDir 'NeuroSynthTerms.mat'],'sTerm');

% Load reference EPI data
Targ = MRIread([DataDir 'target_' ID '.nii']);

% Load target voxels in the cerebral cortex
load([DataDir 'VsetInfo_' ID '.mat'],'vset_info');
ROI = vset_info.IDs;
vset = ROI(1);
voxelSetForm = [DataDir 'vset_' ID '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');

tdata = tdata(tvoxels);
tdata(find(tdata<0)) = 0;

% Calculate corr. coeff. with reference images in NeuroSynth database
CorrData = [];
for mm = 1:length(sTerm)
    disp(['processing NeuroSynth database No. ' num2str(mm)])
    tTerm = char(sTerm(mm));
    tTerm(strfind(tTerm,' ')) = '_';
    M = MRIread([NeuroSynthDir '/' tTerm '_pFgA_pF=0.50_FDR_0.05.nii.gz']); 
    transM = MRIvol2vol(M,Targ,inv(R)); %Transform from MNI152 to EPI space
    mvol = transM.vol;
    mvol = permute(mvol,[2,1,3]);
    mdata =  reshape(mvol,prod(size(mvol)),1);
    mdata = mdata(tvoxels);   
    CorrData(mm) = corr(mdata,tdata);
end

CorrData(find(isnan(CorrData))) = 0;
[SV SI] = sort(CorrData,'descend');

sResult =[];
% Top 50 terms
for ss =1:50
    sname = char(sTerm(SI(ss)));
    sResult.TopRank{ss} = sname;
end

% Bottom 50 terms
for ss =1:50
    sname = char(sTerm(SI(end+1-ss)));
    sResult.BottomRank{ss} = sname;
end

sResult.Term = sTerm;
sResult.CorrData = CorrData;
save([SaveDir 'GroupPCA_' ID '_PC' num2str(PC) '_NeuroSynth.mat'],'sResult');

