clear all, close all

%% by Dr. YIN WANG at Temple University

fs = filesep; % platform-specific file separator

dir_base = '/Users/yw/Desktop/face_PPI_FC_analysis/PPI/fsl_ppi_1st_45ss';
mask_base = '/Users/yw/Desktop/face_PPI_FC_analysis/masks';
result_base = '/Users/yw/Desktop/face_PPI_FC_analysis/PPI/results';


labels_small = {'V1' 'OFA' 'FFA' 'ATL' 'STS' 'IFG' 'AMG' 'OFC' 'PCC'}; 

name_subj = {'100206' '100307' '100408' '100610' '101006' '101107' '101309' '101410' '102311' '102513'...
      '102816' '103111' '103414' '103515' '103818' '104012' '104416' '104820' '105014' '105115'...
      '105216' '105620' '105923' '106016' '106319' '106521' '107018' '107321' '107422' '108121'...
      '108222' '108323' '108525' '108828' '109123' '110411' '110613' '111009' '111312' '111413'... 
      '111514' '111716' '112112' '112314' '112516'};


PPI_task = 'WM';
PPI_model_type = 'full';
FC_folder = 'cope9.feat'; %%% task-state functional connectivity (FC)
EC_folder = 'cope19.feat'; %%% effective connectivity (EC)



k = 1;
% loop through all subjects
%===========================================================================
while (k<=length(name_subj)),
    
    fprintf(1,'==================================\n');
    fprintf(1,'Starting analysis for subject %s\n',name_subj{k});
    fprintf(1,'==================================\n');
    
    %%% seed directory: task-state functional connectivity--- Left/Right Hemisphere
    dir_seed_L_V1_FC  = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-V1_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_OFA_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-OFA_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_FFA_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-FFA_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_ATL_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-ATL_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_STS_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-pSTS_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_IFG_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-IFG_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_AMG_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-AMG_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_OFC_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-OFC_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_L_PCC_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-PCC_',PPI_model_type,'.gfeat' fs FC_folder];
    
    dir_seed_R_V1_FC  = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-V1_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_OFA_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-OFA_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_FFA_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-FFA_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_ATL_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-ATL_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_STS_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-pSTS_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_IFG_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-IFG_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_AMG_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-AMG_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_OFC_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-OFC_',PPI_model_type,'.gfeat' fs FC_folder];
    dir_seed_R_PCC_FC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-PCC_',PPI_model_type,'.gfeat' fs FC_folder];

    %%% seed directory: effective connectivity--- Left/Right Hemisphere
    dir_seed_L_V1_EC  = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-V1_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_OFA_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-OFA_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_FFA_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-FFA_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_ATL_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-ATL_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_STS_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-pSTS_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_IFG_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-IFG_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_AMG_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-AMG_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_OFC_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-OFC_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_L_PCC_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-L-PCC_',PPI_model_type,'.gfeat' fs EC_folder];
    
    dir_seed_R_V1_EC  = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-V1_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_OFA_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-OFA_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_FFA_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-FFA_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_ATL_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-ATL_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_STS_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-pSTS_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_IFG_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-IFG_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_AMG_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-AMG_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_OFC_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-OFC_',PPI_model_type,'.gfeat' fs EC_folder];
    dir_seed_R_PCC_EC = [dir_base fs name_subj{k} fs 'MNINonLinear' fs 'Results' fs 'L2_', PPI_task, '_PPIseed-R-PCC_',PPI_model_type,'.gfeat' fs EC_folder];
    

    %% ROI lists
    ROIS_LH = strvcat([mask_base,fs,name_subj{k},fs,'L_V1.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_OFA.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_FFA.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_ATL.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_pSTS.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_IFG.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_AMG.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_OFC.nii'],...
        [mask_base,fs,name_subj{k},fs,'L_PCC.nii']);
    
    ROIS_RH = strvcat([mask_base,fs,name_subj{k},fs,'R_V1.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_OFA.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_FFA.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_ATL.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_pSTS.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_IFG.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_AMG.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_OFC.nii'],...
        [mask_base,fs,name_subj{k},fs,'R_PCC.nii']);

    %% FC zstats maps for each seed region
    SOURCES_niigz_FC_LH = {[dir_seed_L_V1_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_OFA_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_FFA_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_ATL_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_STS_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_IFG_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_AMG_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_OFC_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_PCC_FC, fs, 'stats' fs 'zstat1.nii.gz']};
 
    SOURCES_niigz_FC_RH = {[dir_seed_R_V1_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_OFA_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_FFA_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_ATL_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_STS_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_IFG_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_AMG_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_OFC_FC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_PCC_FC, fs, 'stats' fs 'zstat1.nii.gz']};
    
    %% EC zstats maps for each seed region
    
    SOURCES_niigz_EC_LH = {[dir_seed_L_V1_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_OFA_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_FFA_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_ATL_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_STS_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_IFG_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_AMG_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_OFC_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_L_PCC_EC, fs, 'stats' fs 'zstat1.nii.gz']};
    
    SOURCES_niigz_EC_RH = {[dir_seed_R_V1_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_OFA_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_FFA_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_ATL_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_STS_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_IFG_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_AMG_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_OFC_EC, fs, 'stats' fs 'zstat1.nii.gz']
        [dir_seed_R_PCC_EC, fs, 'stats' fs 'zstat1.nii.gz']};
    
    %% we have to uncompress these nii.gz files first, so that rex function can read them
    for i=1:9
        gunzip(SOURCES_niigz_FC_LH{i});
        gunzip(SOURCES_niigz_FC_RH{i});
        gunzip(SOURCES_niigz_EC_LH{i});
        gunzip(SOURCES_niigz_EC_RH{i});
    end
    
    %%% below are the names for uncompressed nii and they are ready to be called by the rex function
    SOURCES_nii_FC_LH = {[dir_seed_L_V1_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_OFA_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_FFA_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_ATL_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_STS_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_IFG_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_AMG_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_OFC_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_PCC_FC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_nii_FC_RH = {[dir_seed_R_V1_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_OFA_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_FFA_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_ATL_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_STS_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_IFG_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_AMG_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_OFC_FC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_PCC_FC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_nii_EC_LH = {[dir_seed_L_V1_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_OFA_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_FFA_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_ATL_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_STS_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_IFG_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_AMG_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_OFC_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_L_PCC_EC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_nii_EC_RH = {[dir_seed_R_V1_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_OFA_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_FFA_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_ATL_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_STS_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_IFG_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_AMG_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_OFC_EC, fs, 'stats' fs 'zstat1.nii']
        [dir_seed_R_PCC_EC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_FC_LH = strvcat(SOURCES_nii_FC_LH{:});
    SOURCES_FC_RH = strvcat(SOURCES_nii_FC_RH{:});
    SOURCES_EC_LH = strvcat(SOURCES_nii_EC_LH{:});
    SOURCES_EC_RH = strvcat(SOURCES_nii_EC_RH{:});
    
    %% Run main function to calculate functional connectivity matrices
    FC_LH = rex(SOURCES_FC_LH, ROIS_LH,'summary_measure','mean','level','rois','gui',0);
    FC_RH = rex(SOURCES_FC_RH, ROIS_RH,'summary_measure','mean','level','rois','gui',0);
    EC_LH = rex(SOURCES_EC_LH, ROIS_LH,'summary_measure','mean','level','rois','gui',0);
    EC_RH = rex(SOURCES_EC_RH, ROIS_RH,'summary_measure','mean','level','rois','gui',0);
    
    %%% after getting results, we need to delete these uncompressed nii files to save some space  
    for i=1:9
        delete(SOURCES_nii_FC_LH{i});
        delete(SOURCES_nii_FC_RH{i});
        delete(SOURCES_nii_EC_LH{i});
        delete(SOURCES_nii_EC_RH{i});
    end
    
    %%%% to make all diagnal elements (self-connection) NaN, see the approach (e.g. linear indexing) below
    %%%% A(1:n+1:end) = v;  %(where v is an n-element vector and n is the number of rows of A). 
    %%%% So, for example, A(1:n+1:end) = diag(B)   %copies the diagonal of B into A. 
    %%%% since we have 9 ROIs, the conversion is: A(1:10:end)= NaN
    FC_LH_diagNaN = FC_LH; FC_LH_diagNaN(1:10:end) = NaN;
    FC_RH_diagNaN = FC_RH; FC_RH_diagNaN(1:10:end) = NaN;
    EC_LH_diagNaN = EC_LH; EC_LH_diagNaN(1:10:end) = NaN;
    EC_RH_diagNaN = EC_RH; EC_RH_diagNaN(1:10:end) = NaN;
    
    %%% making connectivity matrix symmetric (but we should not this symmetric one for effective connectivity because it is directional
    FC_LH_symmetric = (FC_LH_diagNaN + FC_LH_diagNaN')/2;
    FC_RH_symmetric = (FC_RH_diagNaN + FC_RH_diagNaN')/2;
    EC_LH_symmetric = (EC_LH_diagNaN + EC_LH_diagNaN')/2;  %%% should not be used because PPI-EC is directional
    EC_RH_symmetric = (EC_RH_diagNaN + EC_RH_diagNaN')/2;  %%% should not be used because PPI-EC is directional
    
    
    %% save all results
    
    %%%% SAVE original data (diagnal ones were not NaN)
    PPIdata.subjID(k,1) = str2num(name_subj{k});
    PPIdata.FC_LH(:,:,k) = FC_LH;
    PPIdata.FC_RH(:,:,k) = FC_RH;
    PPIdata.EC_LH(:,:,k) = EC_LH;
    PPIdata.EC_RH(:,:,k) = EC_RH;
    
    %%%% SAVE secondary data where diagnal ones were assigned to NaN, 
    PPIdata.FC_LH_diagNaN(:,:,k) = FC_LH_diagNaN;
    PPIdata.FC_RH_diagNaN(:,:,k) = FC_RH_diagNaN;
    PPIdata.EC_LH_diagNaN(:,:,k) = EC_LH_diagNaN;  %%% this non-symetric EC matic should be used
    PPIdata.EC_RH_diagNaN(:,:,k) = EC_RH_diagNaN;  %%% this non-symetric EC matic should be used
    
    %%%% and the whole matrices were converted to be symmetric
    %%%% SAVE the final FC matrices for each hemisphere
    %%%% to calculate the mean FC matrics across subjects, you can just do:
    %%%% mean(FCdata.FC_LH_symmetric,3)
    PPIdata.FC_LH_symmetric(:,:,k) = FC_LH_symmetric; 
    PPIdata.FC_RH_symmetric(:,:,k) = FC_RH_symmetric;
    PPIdata.EC_LH_symmetric(:,:,k) = EC_LH_symmetric;  %%% should not be used because PPI-EC is directional
    PPIdata.EC_RH_symmetric(:,:,k) = EC_RH_symmetric;  %%% should not be used because PPI-EC is directional
    
    %%% save as excel spreadsheet
    PPIdata.FC_LH_excel(k,:) = FC_LH(:)';
    PPIdata.FC_RH_excel(k,:) = FC_RH(:)';
    PPIdata.EC_LH_excel(k,:) = EC_LH(:)';
    PPIdata.EC_RH_excel(k,:) = EC_RH(:)';
    PPIdata.FC_LH_diagNaN_excel(k,:) = FC_LH_diagNaN(:)';
    PPIdata.FC_RH_diagNaN_excel(k,:) = FC_RH_diagNaN(:)';
    PPIdata.EC_LH_diagNaN_excel(k,:) = EC_LH_diagNaN(:)';  
    PPIdata.EC_RH_diagNaN_excel(k,:) = EC_RH_diagNaN(:)';
    PPIdata.FC_LH_symmetric_excel(k,:) = FC_LH_symmetric(:)'; 
    PPIdata.FC_RH_symmetric_excel(k,:) = FC_RH_symmetric(:)';
    PPIdata.EC_LH_symmetric_excel(k,:) = EC_LH_symmetric(:)';  
    PPIdata.EC_RH_symmetric_excel(k,:) = EC_RH_symmetric(:)';

    %%% clear variable for each subjects
    clear dir_seed_L_V1_FC dir_seed_L_OFA_FC dir_seed_L_FFA_FC dir_seed_L_ATL_FC dir_seed_L_STS_FC dir_seed_L_IFG_FC dir_seed_L_AMG_FC dir_seed_L_OFC_FC dir_seed_L_PCC_FC
    clear dir_seed_R_V1_FC dir_seed_R_OFA_FC dir_seed_R_FFA_FC dir_seed_R_ATL_FC dir_seed_R_STS_FC dir_seed_R_IFG_FC dir_seed_R_AMG_FC dir_seed_R_OFC_FC dir_seed_R_PCC_FC
    clear dir_seed_L_V1_EC dir_seed_L_OFA_EC dir_seed_L_FFA_EC dir_seed_L_ATL_EC dir_seed_L_STS_EC dir_seed_L_IFG_EC dir_seed_L_AMG_EC dir_seed_L_OFC_EC dir_seed_L_PCC_EC
    clear dir_seed_R_V1_EC dir_seed_R_OFA_EC dir_seed_R_FFA_EC dir_seed_R_ATL_EC dir_seed_R_STS_EC dir_seed_R_IFG_EC dir_seed_R_AMG_EC dir_seed_R_OFC_EC dir_seed_R_PCC_EC
    clear ROIS_LH ROIS_RH
    clear SOURCES_niigz_FC_LH SOURCES_niigz_FC_RH SOURCES_niigz_EC_LH SOURCES_niigz_EC_RH
    clear SOURCES_nii_FC_LH SOURCES_nii_FC_RH SOURCES_nii_EC_LH SOURCES_nii_EC_RH
    clear SOURCES_FC_LH SOURCES_FC_RH SOURCES_EC_LH SOURCES_EC_RH
    clear FC_LH FC_RH EC_LH EC_RH
    clear FC_LH_diagNaN FC_RH_diagNaN EC_LH_diagNaN EC_RH_diagNaN
    clear FC_LH_symmetric FC_RH_symmetric EC_LH_symmetric EC_RH_symmetric
    
    
    % Switch to next subject
    %=======================================
    k   = k + 1;
    
end  %%% end of main loop

%%%% need to change the names every time!!
cd(result_base)
save('PPI_fullmodel_1st_45ss.mat', 'PPIdata');

excel_filename = 'PPI_fullmodel_1st_45ss.xlsx';

%%%% save subject ID
xlswrite(excel_filename,PPIdata.subjID,'subjectID');

%%%% save FC data in excel spreadsheet
xlswrite(excel_filename,PPIdata.FC_LH_excel,'FC_LH');
xlswrite(excel_filename,PPIdata.FC_RH_excel,'FC_RH');
xlswrite(excel_filename,PPIdata.FC_LH_diagNaN_excel,'FC_LH_diagNaN');
xlswrite(excel_filename,PPIdata.FC_RH_diagNaN_excel,'FC_RH_diagNaN');
xlswrite(excel_filename,PPIdata.FC_LH_symmetric_excel,'FC_LH_symmetric');
xlswrite(excel_filename,PPIdata.FC_RH_symmetric_excel,'FC_RH_symmetric');

%%%% save EC data in excel spreadsheet
xlswrite(excel_filename,PPIdata.EC_LH_excel,'EC_LH');
xlswrite(excel_filename,PPIdata.EC_RH_excel,'EC_RH');
xlswrite(excel_filename,PPIdata.EC_LH_diagNaN_excel,'EC_LH_diagNaN');
xlswrite(excel_filename,PPIdata.EC_RH_diagNaN_excel,'EC_RH_diagNaN');
xlswrite(excel_filename,PPIdata.EC_LH_symmetric_excel,'EC_LH_symmetric');
xlswrite(excel_filename,PPIdata.EC_RH_symmetric_excel,'EC_RH_symmetric');





%% plot label

%%% plot the averaged FC matrices
figure(1), clf
heatmap(mean(PPIdata.FC_LH_symmetric,3), labels_small, labels_small, '%0.2f',...
    'Colormap', 'jet', 'TickAngle', 45, 'GridLines', ':'); %% if you want decimal, just change to '%0.2f'
figure(2), clf
heatmap(mean(PPIdata.FC_RH_symmetric,3), labels_small, labels_small, '%0.2f',...
    'Colormap', 'jet', 'TickAngle', 45, 'GridLines', ':'); %% if you want decimal, just change to '%0.2f'
figure(3), clf
heatmap(mean(PPIdata.EC_LH_diagNaN,3), labels_small, labels_small, '%0.2f',...
    'Colormap', 'jet', 'TickAngle', 45, 'GridLines', ':'); %% if you want decimal, just change to '%0.2f'
figure(4), clf
heatmap(mean(PPIdata.EC_RH_diagNaN,3), labels_small, labels_small, '%0.2f',...
    'Colormap', 'jet', 'TickAngle', 45, 'GridLines', ':'); %% if you want decimal, just change to '%0.2f'

