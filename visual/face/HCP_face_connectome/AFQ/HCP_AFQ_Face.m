clear all, close all

% this script uses Yeatman's AFQ tool box to extract 20 fiber groups from HCP DWI data 
% (HCP provides minimal preprocessing and you also need to do tensor fitting by FSL before using this script)
% this script requires several existing DWI files: DTI_S0.nii; DTI_L1(to L3).nii; DTI_V1(to V3).nii; T1w_acpc_dc_restore_1.25.nii
% it's also recommended to include DTI_FA, DTI_MD, DTI_MO files, in order to do further analysis (not required by the script per se)
% To run this script, you also need to install jason yeatman's AFQ toolbox, and Brian Wandell's VISTASOFT

%%% this script is created by Dr. Yin Wang from Temple University

%% matlab setting
%%% always put vista toolbox and AFQ toolbox at the top of matlab path
which nanmean  % this is just to test whether vista/AFQ tool box is on top of matlab path, you should see nanmean draw from Vista, not from spm or other toolbox
setpref('VISTA', 'verbose', false); %% set up mrVista wrapper for waitbar (progressiion bar for illustration,some matlab version needs this initial setting)
    
%% subject directories
fs = filesep; % platform-specific file separator

dir_base = [fs 'Users' fs 'yw' fs 'Desktop' fs 'AFQ_analysis']; % directory which contains all subject-specific directory, need to specify! this is only for Mac
% windown user could be: dir_base = 'C:\Users\yw\Desktop\AFQ_test';

name_subj = {'109123' '109830' '110411' '110613' '111009' '111312'...
'111413' '111514' '111716' '112112' '112314' '112516' '112920' '113215' '113619' '113922'...
'114217' '114419' '114621' '114823' '115017' '115320' '115825' '116221' '116524' '116726'...
'117122' '117324' '117930' '118124' '118225' '118528' '118730' '118932' '119126' '119732'}; % need to specify every time


k  = 1;


% loop through all subjects
%===========================================================================
while (k<=length(name_subj)),
    
    fprintf(1,'==================================\n');
    fprintf(1,'Starting analysis for subject %s\n',name_subj{k});
    fprintf(1,'==================================\n');
    
    sub_dir = fullfile(dir_base,name_subj{k});  %%% find the path containing FSL preprocessed nifty data
    
    %% Step 0: Convert FSL preprocessed nifty file to AFQ readable DT6 File
    dtiMakeDt6FromFsl([sub_dir,fs,'DTI_S0.nii.gz'],[sub_dir,fs,'T1w_acpc_dc_restore_1.25.nii.gz'],[sub_dir,fs,'dt6.mat'],[true]); %%%% no need to use T1 image for auto align
    % dtiMakeDt6FromFsl('C:\Users\yw\Desktop\HCP_DTI\100206\DTI_S0.nii.gz','C:\Users\yw\Desktop\HCP_DTI\100206\T1w_acpc_dc_restore_1.25.nii.gz'); %%%% no need to use T1 image for auto align
    close all;
    
    %% AFQ_RUN
    clearvars -except k name_subj sub_dir dir_base fs
    
    [afq]=AFQ_run(sub_dir,[0]); %%%% only one healthy group
    cd(sub_dir);
    save('afq.mat', 'afq'); %  afq structure containing all the results
    
    %% save tract properties
    afq20tract.name = {afq.TractProfiles.name};  % AFQ tract name
    afq20tract.info(1,:) = afq.norms.meanVOLUME;   % volume number
    afq20tract.info(2,:) = afq.norms.meanFA;
    afq20tract.info(3,:) = afq.norms.meanMD;
    afq20tract.info(4,:) = afq.norms.meanRD;
    afq20tract.info(5,:) = afq.norms.meanAD;
    for i=1:20
        if isempty(afq.TractProfiles(i).nfibers)
            afq20tract.info(6,i) = NaN
        else
            afq20tract.info(6,i) = afq.TractProfiles(i).nfibers; % streamline count
        end
    end
    
    save('tract_properties.mat','afq20tract');  % in afqtract.info, colums represent 20 tracts, rows represent VOLUME/FA/MD/RD/AD
 
    clearvars -except k name_subj sub_dir dir_base fs
    
    %% Convert fiber .mat to nifty file
    % fiber group names  
    fiber_name={'Left_Thalamic_Radiation','Right_Thalamic_Radiation','Left_Corticospinal','Right_Corticospinal',...
        'Left_Cingulum_Cingulate','Right_Cingulum_Cingulate','Left_Cingulum_Hippocampus','Right_Cingulum_Hippocampus',...
        'Callosum_Forceps_Major','Callosum_Forceps_Minor','Left_IFOF','Right_IFOF',...
        'Left_ILF','Right_ILF','Left_SLF','Right_SLF',...
        'Left_Uncinate','Right_Uncinate','Left_Arcuate','Right_Arcuate'};
    
        
    [dt,t1] = dtiLoadDt6(fullfile([sub_dir,fs,'dt6.mat'])); % load dt6
    fg = dtiReadFibers(fullfile([sub_dir,fs,'fibers',fs,'MoriGroups_clean_D5_L4.mat']));% load fiber groups in MoriGroups_clean
    
    gunzip(fullfile(sub_dir,fs,'T1w_acpc_dc_restore_1.25.nii.gz'));
    native_voxel_space = spm_vol(fullfile(sub_dir,fs,'T1w_acpc_dc_restore_1.25.nii'));
    
    
    for i=1:20 % 20 fiber groups
        fd = dtiComputeFiberDensityNoGUI(fg, dt.xformToAcpc, size(dt.b0), 1, i);
        cd(fullfile([sub_dir,fs,'fibers']));  % all nifty file will be saved under a folder named fibers
        
        %%% save the density of each fiber group as a nifti image (i.e. fiber counts through each voxel)
        dtiWriteNiftiWrapper(single(fd./max(fd(:))),dt.xformToAcpc,fiber_name{i}); % can be changed to [fiber_name{20},'_binary']
%       gunzip(fullfile([sub_dir,'\fibers\',fiber_name{i},'.nii.gz']));
        
        %%% save each fiber as a binary nifti image
        thresh = 0.0001; % this is arbitary, but the principle is to save all voxels belonging to a fiber
        fdCore = dtiCleanImageMask((fd./max(fd(:))>thresh), 2, 1);
        dtiWriteNiftiWrapper(uint8(fdCore),dt.xformToAcpc,[fiber_name{i},'_binary']);
        
        %%% reslice to its native space voxel size
        gunzip(fullfile([sub_dir,fs,'fibers',fs,fiber_name{i},'_binary.nii.gz']))
        fiber_image = spm_vol(fullfile([sub_dir,fs,'fibers',fs,fiber_name{i},'_binary.nii']));
        spm_reslice([native_voxel_space,fiber_image],struct('interp',0,'which',1,'mean',false)); %% nearest neighbour interpolation
        gzip(fullfile([sub_dir,fs,'fibers',fs,'r',fiber_name{i},'_binary.nii']));
        
        clear fiber_image fd fdCore
    end

    delete(fullfile(sub_dir,fs,'T1w_acpc_dc_restore_1.25.nii'));    % delete('*.nii'); 
    
%     %%% render each tract on a figure and save it
%     for i=1:20 % 20 fiber groups
%         fig_tract = AFQ_RenderFibers(fg(i),'dt',dt);  %%%%% need to modify
%         saveas(fig_tract,fiber_name{i},'png');
%         close all hidden
%     end

    clearvars -except k name_subj sub_dir dir_base fs

    cd(dir_base)
    
    %%% calculating overlaps between probtrack fibers and AFQ fibers
    counting_overlap_voxels_face(dir_base, name_subj{k}, 'probabilistic_tractography_face','all_fiber_mask_face');
    
    cd(dir_base)
    
    % Switch to next subject
    %=======================================
    k   = k + 1;
    
end  %%% end of main loop

