clear all, close all

%% by Dr. YIN WANG at Temple University

fs = filesep; % platform-specific file separator

dir_base = '/Users/yw/Desktop/face_REST_analysis/REST/fsl_rest_1st_28ss';
mask_base = '/Users/yw/Desktop/face_REST_analysis/masks';
result_base = '/Users/yw/Desktop/face_REST_analysis/REST/results';

labels_small = {'V1' 'OFA' 'FFA' 'ATL' 'STS' 'IFG' 'AMG' 'OFC' 'PCC'}; 

name_subj = {'100206' '100307' '100408' '100610' '101006' '101107' '101309' '102311' '102513'...
      '102816' '103111' '103414' '103515' '103818' '104012' '104416' '104820' '105014' '105115'...
      '105216' '105620' '105923' '106016' '106319' '106521' '107018' '107321'};

ROI_V1  = 'cope1.feat'; % cope1 = V1;
ROI_OFA = 'cope2.feat'; % cope2 = OFA
ROI_FFA = 'cope3.feat'; % cope3 = FFA
ROI_ATL = 'cope4.feat'; % cope4 = ATL
ROI_STS = 'cope5.feat'; % cope5 = STS
ROI_IFG = 'cope6.feat'; % cope6 = IFG
ROI_AMG = 'cope7.feat'; % cope7 = AMG
ROI_OFC = 'cope8.feat'; % cope8 = OFC
ROI_PCC = 'cope9.feat'; % cope9 = PCC

REST1_LH = 'L2_REST1_L-hemi.gfeat';
REST2_LH = 'L2_REST2_L-hemi.gfeat';
REST1_RH = 'L2_REST1_R-hemi.gfeat';
REST2_RH = 'L2_REST2_R-hemi.gfeat';


k = 1;
% loop through all subjects
%===========================================================================
while (k<=length(name_subj)),
    
    fprintf(1,'==================================\n');
    fprintf(1,'Starting analysis for subject %s\n',name_subj{k});
    fprintf(1,'==================================\n');
    
    sub_dir_REST1_LH = fullfile(dir_base,name_subj{k},fs,'MNINonLinear',fs,'Results',fs,REST1_LH);  
    sub_dir_REST2_LH = fullfile(dir_base,name_subj{k},fs,'MNINonLinear',fs,'Results',fs,REST2_LH);
    sub_dir_REST1_RH = fullfile(dir_base,name_subj{k},fs,'MNINonLinear',fs,'Results',fs,REST1_RH);
    sub_dir_REST2_RH = fullfile(dir_base,name_subj{k},fs,'MNINonLinear',fs,'Results',fs,REST2_RH);

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

    %% resting-states regression zstats maps for each seed region
    SOURCES_niigz_REST1_LH = {[sub_dir_REST1_LH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_LH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii.gz']};
    
    SOURCES_niigz_REST2_LH = {[sub_dir_REST2_LH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_LH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii.gz']};
    
    SOURCES_niigz_REST1_RH = {[sub_dir_REST1_RH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST1_RH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii.gz']};
    
    SOURCES_niigz_REST2_RH = {[sub_dir_REST2_RH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii.gz']
        [sub_dir_REST2_RH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii.gz']};
    
    %% we have to uncompress these nii.gz files first, so that rex function can read them
    for i=1:9
        gunzip(SOURCES_niigz_REST1_LH{i});
        gunzip(SOURCES_niigz_REST2_LH{i});
        gunzip(SOURCES_niigz_REST1_RH{i});
        gunzip(SOURCES_niigz_REST2_RH{i});
    end
    
    %%% below are the names for uncompressed nii and they are ready to be called by the rex function
    SOURCES_nii_REST1_LH = {[sub_dir_REST1_LH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_LH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_nii_REST2_LH = {[sub_dir_REST2_LH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_LH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_nii_REST1_RH = {[sub_dir_REST1_RH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST1_RH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_nii_REST2_RH = {[sub_dir_REST2_RH,fs, ROI_V1, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_OFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_FFA, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_ATL, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_STS, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_IFG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_AMG, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_OFC, fs, 'stats' fs 'zstat1.nii']
        [sub_dir_REST2_RH,fs, ROI_PCC, fs, 'stats' fs 'zstat1.nii']};
    
    SOURCES_REST1_LH = strvcat(SOURCES_nii_REST1_LH{:});
    SOURCES_REST2_LH = strvcat(SOURCES_nii_REST2_LH{:});
    SOURCES_REST1_RH = strvcat(SOURCES_nii_REST1_RH{:});
    SOURCES_REST2_RH = strvcat(SOURCES_nii_REST2_RH{:});
    
    %% Run main function to calculate functional connectivity matrices
    FC_REST1_LH = rex(SOURCES_REST1_LH, ROIS_LH,'summary_measure','mean','level','rois','gui',0);
    FC_REST2_LH = rex(SOURCES_REST2_LH, ROIS_LH,'summary_measure','mean','level','rois','gui',0);
    FC_REST1_RH = rex(SOURCES_REST1_RH, ROIS_RH,'summary_measure','mean','level','rois','gui',0);
    FC_REST2_RH = rex(SOURCES_REST2_RH, ROIS_RH,'summary_measure','mean','level','rois','gui',0);
    
    %%% after getting results, we need to delete these uncompressed nii files to save some space  
    for i=1:9
        delete(SOURCES_nii_REST1_LH{i});
        delete(SOURCES_nii_REST2_LH{i});
        delete(SOURCES_nii_REST1_RH{i});
        delete(SOURCES_nii_REST2_RH{i});
    end
    
    %%%% to make all diagnal elements (self-connection) NaN, see the approach (e.g. linear indexing) below
    %%%% A(1:n+1:end) = v;  %(where v is an n-element vector and n is the number of rows of A). 
    %%%% So, for example, A(1:n+1:end) = diag(B)   %copies the diagonal of B into A. 
    %%%% since we have 9 ROIs, the conversion is: A(1:10:end)= NaN
    FC_REST1_LH_diagNaN = FC_REST1_LH; FC_REST1_LH_diagNaN(1:10:end) = NaN;
    FC_REST2_LH_diagNaN = FC_REST2_LH; FC_REST2_LH_diagNaN(1:10:end) = NaN;
    FC_REST1_RH_diagNaN = FC_REST1_RH; FC_REST1_RH_diagNaN(1:10:end) = NaN;
    FC_REST2_RH_diagNaN = FC_REST2_RH; FC_REST2_RH_diagNaN(1:10:end) = NaN;
    
    %%% making connectivity matrix symmetric 
    FC_REST1_LH_symmetric = (FC_REST1_LH_diagNaN + FC_REST1_LH_diagNaN')/2;
    FC_REST2_LH_symmetric = (FC_REST2_LH_diagNaN + FC_REST2_LH_diagNaN')/2;
    FC_REST1_RH_symmetric = (FC_REST1_RH_diagNaN + FC_REST1_RH_diagNaN')/2;
    FC_REST2_RH_symmetric = (FC_REST2_RH_diagNaN + FC_REST2_RH_diagNaN')/2;
    
    %%% average REST1 and REST2 for each hemisphere
    FC_LH_symmetric = (FC_REST1_LH_symmetric + FC_REST2_LH_symmetric)/2;
    FC_RH_symmetric = (FC_REST1_RH_symmetric + FC_REST2_RH_symmetric)/2;
    
    %% save all results
    
    %%%% SAVE original data (diagnal ones were not NaN)
    RESTdata.subjID(k,1) = str2num(name_subj{k});
    RESTdata.REST1_LH(:,:,k) = FC_REST1_LH;
    RESTdata.REST2_LH(:,:,k) = FC_REST2_LH;
    RESTdata.REST1_RH(:,:,k) = FC_REST1_RH;
    RESTdata.REST2_RH(:,:,k) = FC_REST2_RH;
    
    %%%% SAVE secondary data where diagnal ones were assigned to NaN, 
    RESTdata.REST1_LH_diagNaN(:,:,k) = FC_REST1_LH_diagNaN;
    RESTdata.REST2_LH_diagNaN(:,:,k) = FC_REST2_LH_diagNaN;
    RESTdata.REST1_RH_diagNaN(:,:,k) = FC_REST1_RH_diagNaN;
    RESTdata.REST2_RH_diagNaN(:,:,k) = FC_REST2_RH_diagNaN;
    %%%% and the whole matrices were converted to be symmetric
    RESTdata.REST1_LH_symmetric(:,:,k) = FC_REST1_LH_symmetric;
    RESTdata.REST2_LH_symmetric(:,:,k) = FC_REST2_LH_symmetric;
    RESTdata.REST1_RH_symmetric(:,:,k) = FC_REST1_RH_symmetric;
    RESTdata.REST2_RH_symmetric(:,:,k) = FC_REST2_RH_symmetric;
    
    %%%% SAVE the final FC matrices for each hemisphere
    %%%% to calculate the mean FC matrics across subjects, you can just do:
    %%%% mean(FCdata.LH_symmetric,3); mean(FCdata.RH_symmetric,3)
    RESTdata.LH_symmetric(:,:,k) = FC_LH_symmetric;
    RESTdata.RH_symmetric(:,:,k) = FC_RH_symmetric;
    
    %%% save as excel spreadsheet
    RESTdata.FC_REST1_LH_excel(k,:) = FC_REST1_LH(:)';
    RESTdata.FC_REST2_LH_excel(k,:) = FC_REST2_LH(:)';
    RESTdata.FC_REST1_RH_excel(k,:) = FC_REST1_RH(:)';
    RESTdata.FC_REST2_RH_excel(k,:) = FC_REST2_RH(:)';
    RESTdata.FC_REST1_LH_diagNaN_excel(k,:) = FC_REST1_LH_diagNaN(:)';
    RESTdata.FC_REST2_LH_diagNaN_excel(k,:) = FC_REST2_LH_diagNaN(:)';
    RESTdata.FC_REST1_RH_diagNaN_excel(k,:) = FC_REST1_RH_diagNaN(:)';  
    RESTdata.FC_REST2_RH_diagNaN_excel(k,:) = FC_REST2_RH_diagNaN(:)';
    RESTdata.FC_REST1_LH_symmetric_excel(k,:) = FC_REST1_LH_symmetric(:)'; 
    RESTdata.FC_REST2_LH_symmetric_excel(k,:) = FC_REST2_LH_symmetric(:)';
    RESTdata.FC_REST1_RH_symmetric_excel(k,:) = FC_REST1_RH_symmetric(:)';  
    RESTdata.FC_REST2_RH_symmetric_excel(k,:) = FC_REST2_RH_symmetric(:)';
    RESTdata.FC_LH_symmetric_excel(k,:) = FC_LH_symmetric(:)';  
    RESTdata.FC_RH_symmetric_excel(k,:) = FC_RH_symmetric(:)';
    
    
    %%% clear variable for each subjects
    clear sub_dir_REST1_LH sub_dir_REST2_LH sub_dir_REST1_RH sub_dir_REST2_RH
    clear ROIS_LH ROIS_RH
    clear SOURCES_niigz_REST1_LH SOURCES_niigz_REST2_LH SOURCES_niigz_REST1_RH SOURCES_niigz_REST2_RH
    clear SOURCES_nii_REST1_LH SOURCES_nii_REST2_LH SOURCES_nii_REST1_RH SOURCES_nii_REST2_RH
    clear SOURCES_REST1_LH SOURCES_REST2_LH SOURCES_REST1_RH SOURCES_REST2_RH
    clear FC_REST1_LH FC_REST2_LH FC_REST1_RH FC_REST2_RH
    clear FC_REST1_LH_diagNaN FC_REST2_LH_diagNaN FC_REST1_RH_diagNaN FC_REST2_RH_diagNaN
    clear FC_REST1_LH_symmetric FC_REST2_LH_symmetric FC_REST1_RH_symmetric FC_REST2_RH_symmetric
    clear FC_LH_symmetric FC_RH_symmetric
    
     
    
    % Switch to next subject
    %=======================================
    k   = k + 1;
    
end  %%% end of main loop

cd(result_base)
save('REST_1st_28ss.mat', 'RESTdata');

excel_filename = 'REST_1st_28ss.xlsx';

%%%% save subject ID
xlswrite(excel_filename,RESTdata.subjID,'subjectID');
xlswrite(excel_filename,RESTdata.FC_REST1_LH_excel,'FC_REST1_LH');
xlswrite(excel_filename,RESTdata.FC_REST2_LH_excel,'FC_REST2_LH');
xlswrite(excel_filename,RESTdata.FC_REST1_RH_excel,'FC_REST1_RH');
xlswrite(excel_filename,RESTdata.FC_REST2_RH_excel,'FC_REST2_RH');
xlswrite(excel_filename,RESTdata.FC_REST1_LH_diagNaN_excel,'FC_REST1_LH_diagNaN');
xlswrite(excel_filename,RESTdata.FC_REST2_LH_diagNaN_excel,'FC_REST2_LH_diagNaN');
xlswrite(excel_filename,RESTdata.FC_REST1_RH_diagNaN_excel,'FC_REST1_RH_diagNaN');
xlswrite(excel_filename,RESTdata.FC_REST2_RH_diagNaN_excel,'FC_REST2_RH_diagNaN');
xlswrite(excel_filename,RESTdata.FC_REST1_LH_symmetric_excel,'FC_REST1_LH_symmetric');
xlswrite(excel_filename,RESTdata.FC_REST2_LH_symmetric_excel,'FC_REST2_LH_symmetric');
xlswrite(excel_filename,RESTdata.FC_REST1_RH_symmetric_excel,'FC_REST1_RH_symmetric');
xlswrite(excel_filename,RESTdata.FC_REST2_RH_symmetric_excel,'FC_REST2_RH_symmetric');
xlswrite(excel_filename,RESTdata.FC_LH_symmetric_excel,'FC_LH_symmetric');
xlswrite(excel_filename,RESTdata.FC_RH_symmetric_excel,'FC_RH_symmetric');



%% plot label

%%% plot the averaged FC matrices
figure(1), clf
heatmap(mean(RESTdata.LH_symmetric,3), labels_small, labels_small, '%0.2f',...
    'Colormap', 'jet', 'TickAngle', 45, 'GridLines', ':'); %% if you want decimal, just change to '%0.2f'
figure(2), clf
heatmap(mean(RESTdata.RH_symmetric,3), labels_small, labels_small, '%0.2f',...
    'Colormap', 'jet', 'TickAngle', 45, 'GridLines', ':'); %% if you want decimal, just change to '%0.2f'

