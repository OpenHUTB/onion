function counting_overlap_voxels_face(dir_base, name_subj,probabilistic_tractography,all_fiber_mask)
% This function will calculate the voxel overlap between FSL-probtrack fibers and AFQ fibers
% INPUT
% dir_base: the directory level containng all subjects folder, e.g. dir_base = 'C:\Users\yw\Desktop\AFQ_test';
% name_subj: the current subject folder, e.g. name_subj = '100206';
% probabilistic_tractography: the folder name storing all results from FSL probtrack. e.g. 'probabilistic_tractography_face';
% all_fiber_mask: a new folder will be created by this function where all probtrack and AFQ fiber masks will be stored. e.g.  'all_fiber_mask_face'; 
% OUTPUT
% this function will create a mat file called 'AFQ_Probtrax_overlap_data.mat' in 'all_fiber_mask' folder
% 
% To run this script, you also need to install jason yeatman's AFQ toolbox, and Brian Wandell's VISTASOFT
% This script is created by Dr. Yin Wang from Temple University



%%% following is just for testing this function, DON'T USE
% dir_base = 'C:\Users\yw\Desktop\AFQ_test';
% name_subj = '100206'; % need to specify every time
% probabilistic_tractography = 'probabilistic_tractography_face';
% all_fiber_mask = 'all_fiber_mask_face';

fs = filesep; % platform-specific file separator

sub_dir = fullfile(dir_base,name_subj);
cd(sub_dir); mkdir(all_fiber_mask); % all fiber masks will be saved here
fiber_dir = fullfile(sub_dir,all_fiber_mask); %% Where we will do all fiber overlap percentage analysis

%% copy probabilistic_tractography masks to all_fiber_mask folder 
cd(fullfile(sub_dir,probabilistic_tractography));

%get all track names  %%% need to change if its a unilateral region such as VTA
a=dir('L*'); b=dir('R*'); 
probtrack_names = [{a.name},{b.name}]'; 
clear a b
% probtrack_names = [ls('L*');ls('R*')]; % this is for windows system 

for i=1: size(probtrack_names,1)
    cd(fullfile(sub_dir,probabilistic_tractography,probtrack_names{i,:}));
    oldfilename = ['fdt_paths_native_sd_thr01_bin.nii.gz'];  %%% can be changed to 5% percentage threshold (P5/P25)
    newfilename = [fiber_dir,fs,probtrack_names{i,:},'_',oldfilename];
    copyfile(oldfilename,newfilename);
end

%% copy AFQ output (reslieced, r-prefix) fiber masks to all_fiber_mask folder 
cd(fullfile(sub_dir,'fibers'));
copyfile('rLeft*_binary.nii.gz',fiber_dir);
copyfile('rRight*_binary.nii.gz',fiber_dir);
copyfile('rCallosum*_binary.nii.gz',fiber_dir)


%% fiber names from ProbtrackX and AFQ 
%% afqpt is a domain storing all information for probalistic tractography and afq
afqpt.AFQ_fiber_name_left={'Left_Thalamic_Radiation'
    'Left_Corticospinal'
    'Left_Cingulum_Cingulate'
    'Left_Cingulum_Hippocampus'
    'Callosum_Forceps_Major'
    'Callosum_Forceps_Minor'
    'Left_IFOF'
    'Left_ILF'
    'Left_SLF'
    'Left_Uncinate'
    'Left_Arcuate'};

afqpt.AFQ_fiber_name_right={'Right_Thalamic_Radiation'
    'Right_Corticospinal'
    'Right_Cingulum_Cingulate'
    'Right_Cingulum_Hippocampus'
    'Callosum_Forceps_Major'
    'Callosum_Forceps_Minor'
    'Right_IFOF'
    'Right_ILF'
    'Right_SLF'
    'Right_Uncinate'
    'Right_Arcuate'};

afqpt.probtrackROI_left={'L_V1','L_OFA','L_FFA','L_ATL','L_pSTS','L_IFG','L_AMG','L_OFC','L_PCC','L_PPA'};
afqpt.probtrackROI_right={'R_V1','R_OFA','R_FFA','R_ATL','R_pSTS','R_IFG','R_AMG','R_OFC','R_PCC','R_PPA'};

afqpt.probtract_left = {'L_V1_L_V1'   'L_OFA_L_V1'   'L_FFA_L_V1'   'L_ATL_L_V1'   'L_pSTS_L_V1'   'L_IFG_L_V1'   'L_AMG_L_V1'   'L_OFC_L_V1'   'L_PCC_L_V1'   'L_PPA_L_V1'
'L_OFA_L_V1'  'L_OFA_L_OFA'  'L_FFA_L_OFA'  'L_ATL_L_OFA'  'L_OFA_L_pSTS'  'L_IFG_L_OFA'  'L_AMG_L_OFA'  'L_OFA_L_OFC'  'L_OFA_L_PCC'  'L_OFA_L_PPA'                                                             
'L_FFA_L_V1'  'L_FFA_L_OFA'  'L_FFA_L_FFA'  'L_ATL_L_FFA'  'L_FFA_L_pSTS'  'L_FFA_L_IFG'  'L_AMG_L_FFA'  'L_FFA_L_OFC'  'L_FFA_L_PCC'  'L_FFA_L_PPA'      
'L_ATL_L_V1'  'L_ATL_L_OFA'  'L_ATL_L_FFA'  'L_ATL_L_ATL'  'L_ATL_L_pSTS'  'L_ATL_L_IFG'  'L_AMG_L_ATL'  'L_ATL_L_OFC'  'L_ATL_L_PCC'  'L_ATL_L_PPA'        
'L_pSTS_L_V1' 'L_OFA_L_pSTS' 'L_FFA_L_pSTS' 'L_ATL_L_pSTS' 'L_pSTS_L_pSTS' 'L_IFG_L_pSTS' 'L_AMG_L_pSTS' 'L_OFC_L_pSTS' 'L_PCC_L_pSTS' 'L_PPA_L_pSTS'        
'L_IFG_L_V1'  'L_IFG_L_OFA'  'L_FFA_L_IFG'  'L_ATL_L_IFG'  'L_IFG_L_pSTS'  'L_IFG_L_IFG'  'L_AMG_L_IFG'  'L_IFG_L_OFC'  'L_IFG_L_PCC'  'L_IFG_L_PPA'         
'L_AMG_L_V1'  'L_AMG_L_OFA'  'L_AMG_L_FFA'  'L_AMG_L_ATL'  'L_AMG_L_pSTS'  'L_AMG_L_IFG'  'L_AMG_L_AMG'  'L_AMG_L_OFC'  'L_AMG_L_PCC'  'L_AMG_L_PPA'     
'L_OFC_L_V1'  'L_OFA_L_OFC'  'L_FFA_L_OFC'  'L_ATL_L_OFC'  'L_OFC_L_pSTS'  'L_IFG_L_OFC'  'L_AMG_L_OFC'  'L_OFC_L_OFC'  'L_OFC_L_PCC'  'L_OFC_L_PPA'     
'L_PCC_L_V1'  'L_OFA_L_PCC'  'L_FFA_L_PCC'  'L_ATL_L_PCC'  'L_PCC_L_pSTS'  'L_IFG_L_PCC'  'L_AMG_L_PCC'  'L_OFC_L_PCC'  'L_PCC_L_PCC'  'L_PCC_L_PPA'    
'L_PPA_L_V1'  'L_OFA_L_PPA'  'L_FFA_L_PPA'  'L_ATL_L_PPA'  'L_PPA_L_pSTS'  'L_IFG_L_PPA'  'L_AMG_L_PPA'  'L_OFC_L_PPA'  'L_PCC_L_PPA'  'L_PPA_L_PPA'};

afqpt.probtract_right = {'R_V1_R_V1'   'R_OFA_R_V1'   'R_FFA_R_V1'   'R_ATL_R_V1'   'R_pSTS_R_V1'   'R_IFG_R_V1'   'R_AMG_R_V1'   'R_OFC_R_V1'   'R_PCC_R_V1'   'R_PPA_R_V1'
'R_OFA_R_V1'  'R_OFA_R_OFA'  'R_FFA_R_OFA'  'R_ATL_R_OFA'  'R_OFA_R_pSTS'  'R_IFG_R_OFA'  'R_AMG_R_OFA'  'R_OFA_R_OFC'  'R_OFA_R_PCC'  'R_OFA_R_PPA'                                                             
'R_FFA_R_V1'  'R_FFA_R_OFA'  'R_FFA_R_FFA'  'R_ATL_R_FFA'  'R_FFA_R_pSTS'  'R_FFA_R_IFG'  'R_AMG_R_FFA'  'R_FFA_R_OFC'  'R_FFA_R_PCC'  'R_FFA_R_PPA'      
'R_ATL_R_V1'  'R_ATL_R_OFA'  'R_ATL_R_FFA'  'R_ATL_R_ATL'  'R_ATL_R_pSTS'  'R_ATL_R_IFG'  'R_AMG_R_ATL'  'R_ATL_R_OFC'  'R_ATL_R_PCC'  'R_ATL_R_PPA'        
'R_pSTS_R_V1' 'R_OFA_R_pSTS' 'R_FFA_R_pSTS' 'R_ATL_R_pSTS' 'R_pSTS_R_pSTS' 'R_IFG_R_pSTS' 'R_AMG_R_pSTS' 'R_OFC_R_pSTS' 'R_PCC_R_pSTS' 'R_PPA_R_pSTS'        
'R_IFG_R_V1'  'R_IFG_R_OFA'  'R_FFA_R_IFG'  'R_ATL_R_IFG'  'R_IFG_R_pSTS'  'R_IFG_R_IFG'  'R_AMG_R_IFG'  'R_IFG_R_OFC'  'R_IFG_R_PCC'  'R_IFG_R_PPA'         
'R_AMG_R_V1'  'R_AMG_R_OFA'  'R_AMG_R_FFA'  'R_AMG_R_ATL'  'R_AMG_R_pSTS'  'R_AMG_R_IFG'  'R_AMG_R_AMG'  'R_AMG_R_OFC'  'R_AMG_R_PCC'  'R_AMG_R_PPA'     
'R_OFC_R_V1'  'R_OFA_R_OFC'  'R_FFA_R_OFC'  'R_ATL_R_OFC'  'R_OFC_R_pSTS'  'R_IFG_R_OFC'  'R_AMG_R_OFC'  'R_OFC_R_OFC'  'R_OFC_R_PCC'  'R_OFC_R_PPA'     
'R_PCC_R_V1'  'R_OFA_R_PCC'  'R_FFA_R_PCC'  'R_ATL_R_PCC'  'R_PCC_R_pSTS'  'R_IFG_R_PCC'  'R_AMG_R_PCC'  'R_OFC_R_PCC'  'R_PCC_R_PCC'  'R_PCC_R_PPA'    
'R_PPA_R_V1'  'R_OFA_R_PPA'  'R_FFA_R_PPA'  'R_ATL_R_PPA'  'R_PPA_R_pSTS'  'R_IFG_R_PPA'  'R_AMG_R_PPA'  'R_OFC_R_PPA'  'R_PCC_R_PPA'  'R_PPA_R_PPA'};   

%%% final output percentage matrix will be something like this
%             L_V1'        'L_OFA'        'L_FFA'        'L_ATL'        'L_pSTS'        'L_IFG'        'L_AMG'        'L_OFC'        'L_PCC'        'L_PPA'
% 'L_V1'   'L_V1_L_V1'   'L_OFA_L_V1'   'L_FFA_L_V1'   'L_ATL_L_V1'   'L_pSTS_L_V1'   'L_IFG_L_V1'   'L_AMG_L_V1'   'L_OFC_L_V1'   'L_PCC_L_V1'   'L_PPA_L_V1'
% 'L_OFA'  'L_OFA_L_V1'  'L_OFA_L_OFA'  'L_FFA_L_OFA'  'L_ATL_L_OFA'  'L_OFA_L_pSTS'  'L_IFG_L_OFA'  'L_AMG_L_OFA'  'L_OFA_L_OFC'  'L_OFA_L_PCC'  'L_OFA_L_PPA'                                                             
% 'L_FFA'  'L_FFA_L_V1'  'L_FFA_L_OFA'  'L_FFA_L_FFA'  'L_ATL_L_FFA'  'L_FFA_L_pSTS'  'L_FFA_L_IFG'  'L_AMG_L_FFA'  'L_FFA_L_OFC'  'L_FFA_L_PCC'  'L_FFA_L_PPA'      
% 'L_ATL'  'L_ATL_L_V1'  'L_ATL_L_OFA'  'L_ATL_L_FFA'  'L_ATL_L_ATL'  'L_ATL_L_pSTS'  'L_ATL_L_IFG'  'L_AMG_L_ATL'  'L_ATL_L_OFC'  'L_ATL_L_PCC'  'L_ATL_L_PPA'        
% 'L_pSTS' 'L_pSTS_L_V1' 'L_OFA_L_pSTS' 'L_FFA_L_pSTS' 'L_ATL_L_pSTS' 'L_pSTS_L_pSTS' 'L_IFG_L_pSTS' 'L_AMG_L_pSTS' 'L_OFC_L_pSTS' 'L_PCC_L_pSTS' 'L_PPA_L_pSTS'        
% 'L_IFG'  'L_IFG_L_V1'  'L_IFG_L_OFA'  'L_FFA_L_IFG'  'L_ATL_L_IFG'  'L_IFG_L_pSTS'  'L_IFG_L_IFG'  'L_AMG_L_IFG'  'L_IFG_L_OFC'  'L_IFG_L_PCC'  'L_IFG_L_PPA'         
% 'L_AMG'  'L_AMG_L_V1'  'L_AMG_L_OFA'  'L_AMG_L_FFA'  'L_AMG_L_ATL'  'L_AMG_L_pSTS'  'L_AMG_L_IFG'  'L_AMG_L_AMG'  'L_AMG_L_OFC'  'L_AMG_L_PCC'  'L_AMG_L_PPA'     
% 'L_OFC'  'L_OFC_L_V1'  'L_OFA_L_OFC'  'L_FFA_L_OFC'  'L_ATL_L_OFC'  'L_OFC_L_pSTS'  'L_IFG_L_OFC'  'L_AMG_L_OFC'  'L_OFC_L_OFC'  'L_OFC_L_PCC'  'L_OFC_L_PPA'     
% 'L_PCC'  'L_PCC_L_V1'  'L_OFA_L_PCC'  'L_FFA_L_PCC'  'L_ATL_L_PCC'  'L_PCC_L_pSTS'  'L_IFG_L_PCC'  'L_AMG_L_PCC'  'L_OFC_L_PCC'  'L_PCC_L_PCC'  'L_PCC_L_PPA'    
% 'L_PPA'  'L_PPA_L_V1'  'L_OFA_L_PPA'  'L_FFA_L_PPA'  'L_ATL_L_PPA'  'L_PPA_L_pSTS'  'L_IFG_L_PPA'  'L_AMG_L_PPA'  'L_OFC_L_PPA'  'L_PCC_L_PPA'  'L_PPA_L_PPA'    


%             R_V1'        'R_OFA'        'R_FFA'        'R_ATL'        'R_pSTS'        'R_IFG'        'R_AMG'        'R_OFC'        'R_PCC'        'R_PPA'
% 'R_V1'   'R_V1_R_V1'   'R_OFA_R_V1'   'R_FFA_R_V1'   'R_ATL_R_V1'   'R_pSTS_R_V1'   'R_IFG_R_V1'   'R_AMG_R_V1'   'R_OFC_R_V1'   'R_PCC_R_V1'   'R_PPA_R_V1'
% 'R_OFA'  'R_OFA_R_V1'  'R_OFA_R_OFA'  'R_FFA_R_OFA'  'R_ATL_R_OFA'  'R_OFA_R_pSTS'  'R_IFG_R_OFA'  'R_AMG_R_OFA'  'R_OFA_R_OFC'  'R_OFA_R_PCC'  'R_OFA_R_PPA'                                                             
% 'R_FFA'  'R_FFA_R_V1'  'R_FFA_R_OFA'  'R_FFA_R_FFA'  'R_ATL_R_FFA'  'R_FFA_R_pSTS'  'R_FFA_R_IFG'  'R_AMG_R_FFA'  'R_FFA_R_OFC'  'R_FFA_R_PCC'  'R_FFA_R_PPA'      
% 'R_ATL'  'R_ATL_R_V1'  'R_ATL_R_OFA'  'R_ATL_R_FFA'  'R_ATL_R_ATL'  'R_ATL_R_pSTS'  'R_ATL_R_IFG'  'R_AMG_R_ATL'  'R_ATL_R_OFC'  'R_ATL_R_PCC'  'R_ATL_R_PPA'        
% 'R_pSTS' 'R_pSTS_R_V1' 'R_OFA_R_pSTS' 'R_FFA_R_pSTS' 'R_ATL_R_pSTS' 'R_pSTS_R_pSTS' 'R_IFG_R_pSTS' 'R_AMG_R_pSTS' 'R_OFC_R_pSTS' 'R_PCC_R_pSTS' 'R_PPA_R_pSTS'        
% 'R_IFG'  'R_IFG_R_V1'  'R_IFG_R_OFA'  'R_FFA_R_IFG'  'R_ATL_R_IFG'  'R_IFG_R_pSTS'  'R_IFG_R_IFG'  'R_AMG_R_IFG'  'R_IFG_R_OFC'  'R_IFG_R_PCC'  'R_IFG_R_PPA'         
% 'R_AMG'  'R_AMG_R_V1'  'R_AMG_R_OFA'  'R_AMG_R_FFA'  'R_AMG_R_ATL'  'R_AMG_R_pSTS'  'R_AMG_R_IFG'  'R_AMG_R_AMG'  'R_AMG_R_OFC'  'R_AMG_R_PCC'  'R_AMG_R_PPA'     
% 'R_OFC'  'R_OFC_R_V1'  'R_OFA_R_OFC'  'R_FFA_R_OFC'  'R_ATL_R_OFC'  'R_OFC_R_pSTS'  'R_IFG_R_OFC'  'R_AMG_R_OFC'  'R_OFC_R_OFC'  'R_OFC_R_PCC'  'R_OFC_R_PPA'     
% 'R_PCC'  'R_PCC_R_V1'  'R_OFA_R_PCC'  'R_FFA_R_PCC'  'R_ATL_R_PCC'  'R_PCC_R_pSTS'  'R_IFG_R_PCC'  'R_AMG_R_PCC'  'R_OFC_R_PCC'  'R_PCC_R_PCC'  'R_PCC_R_PPA'    
% 'R_PPA'  'R_PPA_R_V1'  'R_OFA_R_PPA'  'R_FFA_R_PPA'  'R_ATL_R_PPA'  'R_PPA_R_pSTS'  'R_IFG_R_PPA'  'R_AMG_R_PPA'  'R_OFC_R_PPA'  'R_PCC_R_PPA'  'R_PPA_R_PPA'   


%% unzip all fiber masks so that we can use the function load_nii
cd(fiber_dir);
gunzip('*.nii.gz'); 

%%% left hemisphere

for i=1:size(afqpt.probtrackROI_left,2)
    
    for j=1:size(afqpt.probtrackROI_left,2)
        
        if ~isempty(dir(['*',afqpt.probtract_left{i,j},'*.nii'])); %% if the probtract file exist (some subjects don't have one or two ROIs, so they don't have that ROI-pairwise tract)
            afqpt.ProbtrackX_tract_exist_LH(i,j) = 1;
            probtract_file_LH = dir(['*',afqpt.probtract_left{i,j},'*.nii']);
            ProbtrackX_fiber_images_LH = load_nii(probtract_file_LH.name);
            afqpt.count_ProbtrackX_voxels_LH(i,j) = sum(ProbtrackX_fiber_images_LH.img(:));  %%% the voxel number within the probatrack fibers
            
            for k=1:size(afqpt.AFQ_fiber_name_left,1)
                
                if ~isempty(dir(['*',afqpt.AFQ_fiber_name_left{k},'*.nii'])); %% if the AFQ tract file exist (AFQ might not be able to get all tracts for every subject)
                    afqpt.AFQ_tract_exist_LH(i,j,k) = 1;
                    %%% counting voxels in each fiber
                    AFQ_tract_file_LH = dir(['*',afqpt.AFQ_fiber_name_left{k},'*.nii']);
                    AFQ_fiber_images_LH= load_nii(AFQ_tract_file_LH.name);
                    afqpt.count_AFQ_voxels_LH(i,j,k) = sum(AFQ_fiber_images_LH.img(:));
                    
                    %%% counting overlap voxels between AFQ fiber mask and probabilistic fiber mask
                    overlapImage_LH = AFQ_fiber_images_LH.img & ProbtrackX_fiber_images_LH.img;
                    % To count number of pixels
                    afqpt.numOverlapPixels_LH(i,j,k) = nnz(overlapImage_LH);
                    afqpt.Percentage_OverlapPixels_LH(i,j,k)= afqpt.numOverlapPixels_LH(i,j,k)/afqpt.count_ProbtrackX_voxels_LH(i,j);
                    
                    clear AFQ_tract_file_LH AFQ_fiber_images_LH overlapImage_LH
                    
                else
                    afqpt.AFQ_tract_exist_LH(i,j,k) = 0;
                    afqpt.count_AFQ_voxels_LH(i,j,k) = NaN;
                    afqpt.numOverlapPixels_LH(i,j,k) = NaN;
                    afqpt.Percentage_OverlapPixels_LH(i,j,k) = NaN;
                end

            end
            
        else  %% if the ROI-pairwise tract does not exist
            afqpt.ProbtrackX_tract_exist_LH(i,j) = 0;
            afqpt.count_ProbtrackX_voxels_LH(i,j)= NaN;
     
            for k=1:size(afqpt.AFQ_fiber_name_left,1)
                
                if ~isempty(dir(['*',afqpt.AFQ_fiber_name_left{k},'*.nii'])); %% if the AFQ tract file exist (AFQ might not be able to get all tracts for every subject)
                    afqpt.AFQ_tract_exist_LH(i,j,k) = 1;
                    %%% counting voxels in each fiber
                    AFQ_tract_file_LH = dir(['*',afqpt.AFQ_fiber_name_left{k},'*.nii']);
                    AFQ_fiber_images_LH= load_nii(AFQ_tract_file_LH.name);
                    afqpt.count_AFQ_voxels_LH(i,j,k) = sum(AFQ_fiber_images_LH.img(:)); 
                    afqpt.numOverlapPixels_LH(i,j,k) = NaN;
                    afqpt.Percentage_OverlapPixels_LH(i,j,k)= NaN;
                    
                    clear AFQ_tract_file_LH AFQ_fiber_images_LH
                    
                else
                    afqpt.AFQ_tract_exist_LH(i,j,k) = 0;
                    afqpt.count_AFQ_voxels_LH(i,j,k) = NaN;
                    afqpt.numOverlapPixels_LH(i,j,k) = NaN;
                    afqpt.Percentage_OverlapPixels_LH(i,j,k) = NaN;
                end

            end
        end
        
        clear probtract_file_LH ProbtrackX_fiber_images_LH 
        
    end
end

%%% right hemisphere

for i=1:size(afqpt.probtrackROI_right,2)
    
    for j=1:size(afqpt.probtrackROI_right,2)
        
        if ~isempty(dir(['*',afqpt.probtract_right{i,j},'*.nii'])); %% if the probtract file exist (some subjects don't have one or two ROIs, so they don't have that ROI-pairwise tract)
            afqpt.ProbtrackX_tract_exist_RH(i,j) = 1;
            probtract_file_RH = dir(['*',afqpt.probtract_right{i,j},'*.nii']);
            ProbtrackX_fiber_images_RH = load_nii(probtract_file_RH.name);
            afqpt.count_ProbtrackX_voxels_RH(i,j) = sum(ProbtrackX_fiber_images_RH.img(:));  %%% the voxel number within the probatrack fibers
            
            for k=1:size(afqpt.AFQ_fiber_name_right,1)
                
                if ~isempty(dir(['*',afqpt.AFQ_fiber_name_right{k},'*.nii'])); %% if the AFQ tract file exist (AFQ might not be able to get all tracts for every subject)
                    afqpt.AFQ_tract_exist_RH(i,j,k) = 1;
                    %%% counting voxels in each fiber
                    AFQ_tract_file_RH = dir(['*',afqpt.AFQ_fiber_name_right{k},'*.nii']);
                    AFQ_fiber_images_RH= load_nii(AFQ_tract_file_RH.name);
                    afqpt.count_AFQ_voxels_RH(i,j,k) = sum(AFQ_fiber_images_RH.img(:));
                    
                    %%% counting overlap voxels between AFQ fiber mask and probabilistic fiber mask
                    overlapImage_RH = AFQ_fiber_images_RH.img & ProbtrackX_fiber_images_RH.img;
                    % To count number of pixels
                    afqpt.numOverlapPixels_RH(i,j,k) = nnz(overlapImage_RH);
                    afqpt.Percentage_OverlapPixels_RH(i,j,k)= afqpt.numOverlapPixels_RH(i,j,k)/afqpt.count_ProbtrackX_voxels_RH(i,j);
                    
                    clear AFQ_tract_file_RH AFQ_fiber_images_RH overlapImage_RH
                    
                else
                    afqpt.AFQ_tract_exist_RH(i,j,k) = 0;
                    afqpt.count_AFQ_voxels_RH(i,j,k) = NaN;
                    afqpt.numOverlapPixels_RH(i,j,k) = NaN;
                    afqpt.Percentage_OverlapPixels_RH(i,j,k) = NaN;
                end
            end
            
        else  %% if the ROI-pairwise tract does not exist
            afqpt.ProbtrackX_tract_exist_RH(i,j) = 0;
            afqpt.count_ProbtrackX_voxels_RH(i,j)= NaN;
     
            for k=1:size(afqpt.AFQ_fiber_name_right,1)
                
                if ~isempty(dir(['*',afqpt.AFQ_fiber_name_right{k},'*.nii'])); %% if the AFQ tract file exist (AFQ might not be able to get all tracts for every subject)
                    afqpt.AFQ_tract_exist_RH(i,j,k) = 1;
                    %%% counting voxels in each fiber
                    AFQ_tract_file_RH = dir(['*',afqpt.AFQ_fiber_name_right{k},'*.nii']);
                    AFQ_fiber_images_RH= load_nii(AFQ_tract_file_RH.name);
                    afqpt.count_AFQ_voxels_RH(i,j,k) = sum(AFQ_fiber_images_RH.img(:));
                    afqpt.numOverlapPixels_RH(i,j,k) = NaN;
                    afqpt.Percentage_OverlapPixels_RH(i,j,k)= NaN;
                    
                    clear AFQ_tract_file_RH AFQ_fiber_images_RH 
                    
                else
                    afqpt.AFQ_tract_exist_RH(i,j,k) = 0;
                    afqpt.count_AFQ_voxels_RH(i,j,k) = NaN;
                    afqpt.numOverlapPixels_RH(i,j,k) = NaN;
                    afqpt.Percentage_OverlapPixels_RH(i,j,k) = NaN;
                end
            end
        end
        
        clear probtract_file_RH ProbtrackX_fiber_images_RH 
        
    end
end

delete('*.nii') %% to save some space


save('AFQ_Probtrax_overlap_data.mat', 'afqpt');
% save('AFQ_Probtrax_overlap_data.mat',...   %%% file name saving all overlap information
% 'afqpt.Percentage_OverlapPixels_RH',...  %%% RIGHT HEMESPHERE the percentage of voxels in the probtrack fibers that overlaps with AFQ major fibers
% 'afqpt.numOverlapPixels_RH',...   %%% RIGHT HEMESPHERE number of voxels in the probtrax fibers that overlaps with AFQ major fibers
% 'afqpt.count_AFQ_voxels_RH',...   %%% total number of voxels in the AFQ major fibers
% 'afqpt.count_ProbtrackX_voxels_RH',...  %%%%%% total number of voxels within the probatrack fibers
% 'afqpt.ProbtrackX_tract_exist_RH',...
% 'afqpt.AFQ_tract_exist_RH',...
% 'afqpt.Percentage_OverlapPixels_LH',...
% 'afqpt.numOverlapPixels_LH',...
% 'afqpt.count_AFQ_voxels_LH',...
% 'afqpt.count_ProbtrackX_voxels_LH',...
% 'afqpt.ProbtrackX_tract_exist_LH',...
% 'afqpt.AFQ_tract_exist_LH',...
% 'afqpt.AFQ_fiber_name_left',...
% 'afqpt.AFQ_fiber_name_right',...
% 'afqpt.probtrackROI_left',...
% 'afqpt.probtrackROI_right'...
% 'afqpt.probtrack_left',...
% 'afqpt.probtrack_right');


% visualization of percentage matrix

for i=1:size(afqpt.AFQ_fiber_name_left,1)
    if (max(max(afqpt.Percentage_OverlapPixels_LH(:,:,i)))~= min(min(afqpt.Percentage_OverlapPixels_LH(:,:,i)))) & (max(max(afqpt.AFQ_tract_exist_LH(:,:,i)))==1) %% tract must exist
        h1=HeatMap(afqpt.Percentage_OverlapPixels_LH(:,:,i), 'RowLabels',afqpt.probtrackROI_left,'ColumnLabels', afqpt.probtrackROI_left, 'Standardize',3 ,'Colormap',jet, 'Annotate','on');
        addTitle(h1,afqpt.AFQ_fiber_name_left(i)); %%% e.g. 'Left_IFOF'
        h1Fig = plot(h1);
        saveas(h1Fig,afqpt.AFQ_fiber_name_left{i},'png');
        close all hidden
    else if max(max(afqpt.AFQ_tract_exist_LH(:,:,i)))==1   %%% if all overlap elements are zero (no overlapping), we have to do some tricks to still plot the heatmap 
        replace_matrix = afqpt.Percentage_OverlapPixels_LH(:,:,i);
        replace_matrix(1,2) = 0.0000000000000000000001;  %%% this trick will not be saved, but just for producing a heatmap.png
        h1=HeatMap(replace_matrix, 'RowLabels',afqpt.probtrackROI_left,'ColumnLabels', afqpt.probtrackROI_left, 'Standardize',3 ,'Colormap',jet, 'Annotate','on');
        addTitle(h1,afqpt.AFQ_fiber_name_left(i)); %%% e.g. 'Left_IFOF'
        h1Fig = plot(h1);
        saveas(h1Fig,afqpt.AFQ_fiber_name_left{i},'png');  
        close all hidden
        clear replace_matrix
        end
    end
end

for i=1:size(afqpt.AFQ_fiber_name_right,1)
    if (max(max(afqpt.Percentage_OverlapPixels_RH(:,:,i)))~= min(min(afqpt.Percentage_OverlapPixels_RH(:,:,i)))) & (max(max(afqpt.AFQ_tract_exist_RH(:,:,i)))==1) %% tract must exist
        h1=HeatMap(afqpt.Percentage_OverlapPixels_RH(:,:,i), 'RowLabels',afqpt.probtrackROI_right,'ColumnLabels', afqpt.probtrackROI_right, 'Standardize',3 ,'Colormap',jet, 'Annotate','on');
        addTitle(h1,afqpt.AFQ_fiber_name_right(i)); %%% e.g. 'right_IFOF'
        h1Fig = plot(h1);
        saveas(h1Fig,afqpt.AFQ_fiber_name_right{i},'png');
        close all hidden
    else if max(max(afqpt.AFQ_tract_exist_RH(:,:,i)))==1   %%% if all overlap elements are zero (no overlapping), we have to do some tricks to still plot the heatmap 
        replace_matrix = afqpt.Percentage_OverlapPixels_RH(:,:,i);
        replace_matrix(1,2) = 0.0000000000000000000001;  %%% this trick will not be saved, but just for producing a heatmap.png
        h1=HeatMap(replace_matrix, 'RowLabels',afqpt.probtrackROI_right,'ColumnLabels', afqpt.probtrackROI_right, 'Standardize',3 ,'Colormap',jet, 'Annotate','on');
        addTitle(h1,afqpt.AFQ_fiber_name_right(i)); %%% e.g. 'right_IFOF'
        h1Fig = plot(h1);
        saveas(h1Fig,afqpt.AFQ_fiber_name_right{i},'png'); 
        close all hidden
        clear replace_matrix
        end
    end
end

clear afqpt

