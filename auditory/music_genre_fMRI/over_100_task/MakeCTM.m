function MakeCTM(ID, NeuroSynthDir)
%
% Use other five subjects' data to make neurosynth task-cog. factor transform matrix
% Input: 
% ID... subject ID (e.g., 'sub-01')
% NeuroSynthDir... Directory where reference images in NeuroSynth were saved

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

% Exlude the target subject
Subs = 1:6;
Subs(str2num(ID(6))) = [];

% Calculate correlation btw weight map and reference images for all subjects
tCorrData= [];
Method = 'TaskType';
for ss = 1:length(Subs)
    load([SaveDir 'RidgeResult_' Method '_sub-0' num2str(Subs(ss)) '.mat'], 'Result');    
    w = Result.w;
    tCorrData{ss} = GetCorrData(Subs(ss),w,NeuroSynthDir);            
end

CorrData = zeros(size(tCorrData{1}));
for ss = 1:length(Subs)
    CorrData = CorrData + tCorrData{ss};
end            
CorrData = CorrData / length(Subs);
     
CTM = CorrData;
CTM(find(isnan(CTM))) = 0;
save([SaveDir 'CTM_' ID '.mat'],'CTM');
        


function CorrData = GetCorrData(sID,w,NeuroSynthDir)
    
    DataDir = [pwd '/SampleData/'];
    % Average for three time delayes
    w = squeeze(mean(reshape(w,size(w,1)/3,3,size(w,2)),2));

    % Load registration matrix from  MNI152 space to EPI space
    load([DataDir 'reg_mni152_sub-0' num2str(sID) '.mat'],'R');

    % Load neurosynth terms
    load([DataDir 'NeuroSynthTerms.mat'],'sTerm');

    % Load reference EPI data
    Targ = MRIread([DataDir 'target_sub-0' num2str(sID) '.nii']);

    % Load target voxels in the cerebral cortex
    load([DataDir 'VsetInfo_sub-0' num2str(sID) '.mat'],'vset_info');
    ROI = vset_info.IDs;
    vset = ROI(1); %Specify cerebral cortex voxels
    voxelSetForm = [DataDir 'vset_sub-0' num2str(sID) '/vset_%03d.mat'];
    load(sprintf(voxelSetForm,vset),'tvoxels');

    NSData = [];
    for mm = 1:length(sTerm)
        disp(['processing NeuroSynth database No. ' num2str(mm)])
        tTerm = char(sTerm(mm));
        tTerm(strfind(tTerm,' ')) = '_';
        M = MRIread([NeuroSynthDir '/' tTerm '_pFgA_pF=0.50_FDR_0.05.nii.gz']);
        transM = MRIvol2vol(M,Targ,inv(R));
        mvol = transM.vol;
        mvol = permute(mvol,[2,1,3]);
        mdata =  reshape(mvol,prod(size(mvol)),1);
        mdata = mdata(tvoxels);    
        NSData{mm} = mdata;
    end
    
    % Make TaskType-Neurosynth-transform matrix
    CorrData = [];
    for tt = 1:103
        for mm = 1:length(sTerm)
            CorrData(tt,mm)  = corr(w(tt,:)',NSData{mm});
        end
    end

