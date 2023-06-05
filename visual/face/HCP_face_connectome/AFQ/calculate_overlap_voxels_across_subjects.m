clear all, close all

dir_base = 'E:\HCP_backup\AFQ_completed'; % directory which contains all subject-specific directory, need to specify!
name_subj = {'100206' '100307' '100408' '100610' '101006' '101107' '101309' '101410' '102311' '102513'}; 


fs = filesep; % platform-specific file separator
k  = 1;

% loop through all subjects
%===========================================================================
while (k<=length(name_subj)),
    
    sub_dir = fullfile(dir_base,name_subj{k});
    fiber_dir = fullfile(sub_dir,'all_fiber_mask_face'); %% Where the 'AFQ_Probtrax_overlap_data.mat' is stored
    cd(fiber_dir)
    load AFQ_Probtrax_overlap_data
    
    
    %% left hemisphere
    for i=1:size(afqpt.AFQ_fiber_name_left ,1)
    allsubject.percentage_LH(:,:,i,k)= afqpt.Percentage_OverlapPixels_LH(:,:,i); %%% third column (i) is the tract name, fourth column (k) is for different subjects
    end
    
    
    
    %% right hemisphere
    for i=1:size(afqpt.AFQ_fiber_name_right ,1)
    allsubject.percentage_RH(:,:,i,k)= afqpt.Percentage_OverlapPixels_RH(:,:,i); %%% third column (i) is the tract name, fourth column (k) is for different subjects
    end
    
    allsubject.ID(k)= str2num(name_subj{k});
    
    % Switch to next subject
    %=======================================
    k   = k + 1;
    
end  %%% end of main loop

allsubject.tract_LH = afqpt.AFQ_fiber_name_left;
allsubject.tract_RH = afqpt.AFQ_fiber_name_right;



cd(dir_base)

%%% left hemisphere
for i=1:11
    allsubject.nanmean_LH(:,:,i) = nanmean(allsubject.percentage_LH(:,:,i,:),4);
    h1=HeatMap(allsubject.nanmean_LH(:,:,i), 'RowLabels',afqpt.probtrackROI_left,'ColumnLabels', afqpt.probtrackROI_left, 'Standardize',3 ,'Colormap',jet, 'Annotate','on');
    addTitle(h1,afqpt.AFQ_fiber_name_left(i)); %%% e.g. 'right_IFOF'
    colorbar
    h1Fig = plot(h1);
    saveas(h1Fig,afqpt.AFQ_fiber_name_left{i},'png');
    close all hidden
end

for i=1:10
    for j=1:10
        [B,I]=sort(allsubject.nanmean_LH(i,j,:),'descend');
        newfiber_name = allsubject.tract_LH(I);
        newfiber_percentage = B(:,:)';
        for k=1:11
        allsubject.percentageorder_LH(i,j,k)= {[newfiber_name{k},' = ',num2str(newfiber_percentage(k))]};
        end
        clear B I newfiber_name newfiber_percentage
    end
end





%%% right hemisphere
for i=1:11
    allsubject.nanmean_RH(:,:,i) = nanmean(allsubject.percentage_RH(:,:,i,:),4);
    h2=HeatMap(allsubject.nanmean_RH(:,:,i), 'RowLabels',afqpt.probtrackROI_right,'ColumnLabels', afqpt.probtrackROI_right, 'Standardize',3 ,'Colormap',jet, 'Annotate','on');
    addTitle(h2,afqpt.AFQ_fiber_name_right(i)); %%% e.g. 'right_IFOF'
    colorbar
    h2Fig = plot(h2);
    saveas(h2Fig,afqpt.AFQ_fiber_name_right{i},'bmp');
    close all hidden
end



for i=1:10
    for j=1:10
        [B,I]=sort(allsubject.nanmean_RH(i,j,:),'descend');
        newfiber_name = allsubject.tract_RH(I);
        newfiber_percentage = B(:,:)';
        for k=1:11
        allsubject.percentageorder_RH(i,j,k)= {[newfiber_name{k},' = ',num2str(newfiber_percentage(k))]};
        end
        clear B I newfiber_name newfiber_percentage
    end
end

%%% to search the fiber percentage order for each pair-wise connections, one can type: {allsubject.percentageorder_RH{i,j,:}}' 
% e.g. {allsubject.percentageorder_RH{2,1,:}}' for R_OFA(2) to R_V1(1)
% e.g. {allsubject.percentageorder_RH{2,3,:}}' for R_OFA(2) to R_FFA(3)
% it will give you something like below:
% 'Right_ILF = 0.077645'
% 'Right_IFOF = 0.022861'
% 'Callosum_Forceps_Major = 0.001845'
% 'Right_Arcuate = 0.0017453'
% 'Right_Uncinate = 0.0004717'
% 'Right_Cingulum_Hippocampus = 8.2305e-05'
% 'Right_Thalamic_Radiation = 0'
% 'Right_Corticospinal = 0'
% 'Right_Cingulum_Cingulate = 0'
% 'Callosum_Forceps_Minor = 0'
% 'Right_SLF = 0'

save('mean_overlap_voxels_across_subjects.mat', 'allsubject');

xlswrite('Left_Thalamic_Radiation.xlsx',allsubject.nanmean_LH(:,:,1));
xlswrite('Left_Corticospinal.xlsx',allsubject.nanmean_LH(:,:,2));
xlswrite('Left_Cingulum_Cingulate.xlsx',allsubject.nanmean_LH(:,:,3));
xlswrite('Left_Cingulum_Hippocampus.xlsx',allsubject.nanmean_LH(:,:,4));
xlswrite('Left_Callosum_Forceps_Major.xlsx',allsubject.nanmean_LH(:,:,5));
xlswrite('Left_Callosum_Forceps_Minor.xlsx',allsubject.nanmean_LH(:,:,6));
xlswrite('Left_IFOF.xlsx',allsubject.nanmean_LH(:,:,7));
xlswrite('Left_ILF.xlsx',allsubject.nanmean_LH(:,:,8));
xlswrite('Left_SLF.xlsx',allsubject.nanmean_LH(:,:,9));
xlswrite('Left_Uncinate.xlsx',allsubject.nanmean_LH(:,:,10));
xlswrite('Left_Arcuate.xlsx',allsubject.nanmean_LH(:,:,11));

xlswrite('Right_Thalamic_Radiation.xlsx',allsubject.nanmean_RH(:,:,1));
xlswrite('Right_Corticospinal.xlsx',allsubject.nanmean_RH(:,:,2));
xlswrite('Right_Cingulum_Cingulate.xlsx',allsubject.nanmean_RH(:,:,3));
xlswrite('Right_Cingulum_Hippocampus.xlsx',allsubject.nanmean_RH(:,:,4));
xlswrite('Right_Callosum_Forceps_Major.xlsx',allsubject.nanmean_RH(:,:,5));
xlswrite('Right_Callosum_Forceps_Minor.xlsx',allsubject.nanmean_RH(:,:,6));
xlswrite('Right_IFOF.xlsx',allsubject.nanmean_RH(:,:,7));
xlswrite('Right_ILF.xlsx',allsubject.nanmean_RH(:,:,8));
xlswrite('Right_SLF.xlsx',allsubject.nanmean_RH(:,:,9));
xlswrite('Right_Uncinate.xlsx',allsubject.nanmean_RH(:,:,10));
xlswrite('Right_Arcuate.xlsx',allsubject.nanmean_RH(:,:,11));



