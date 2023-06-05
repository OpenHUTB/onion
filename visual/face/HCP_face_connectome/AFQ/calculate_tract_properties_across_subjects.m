clear all, close all

dir_base = 'E:\HCP_backup\AFQ_completed'; % directory which contains all subject-specific directory, need to specify!
name_subj = {'100206' '100307' '100408' '100610' '101006' '101107' '101309' '101410' '102311' '102513'};

fs = filesep; % platform-specific file separator
k  = 1;

% loop through all subjects
%===========================================================================
while (k<=length(name_subj)),
    
    sub_dir = fullfile(dir_base,name_subj{k}); %% Where the 'tract_properties.mat' is stored
    cd(sub_dir)
    load tract_properties
    
    
    
    %% each column for each fiber
    for i=1:20
        allsubject.meanVOLUME(k,i)= afq20tract.info(1,i); %% row is for each subject, colums are for each of 20 tracts
        allsubject.meanFA(k,i)= afq20tract.info(2,i); %% row is for each subject, colums are for each of 20 tracts
        allsubject.meanMD(k,i)= afq20tract.info(3,i); %% row is for each subject, colums are for each of 20 tracts
        allsubject.meanRD(k,i)= afq20tract.info(4,i); %% row is for each subject, colums are for each of 20 tracts
        allsubject.meanAD(k,i)= afq20tract.info(5,i); %% row is for each subject, colums are for each of 20 tracts
        allsubject.streamline(k,i)= afq20tract.info(6,i); %% row is for each subject, colums are for each of 20 tracts
    end
    
    %     column 1='Left Thalamic Radiation'
    %     column 2='Right Thalamic Radiation'
    %     column 3='Left Corticospinal'
    %     column 4='Right Corticospinal'
    %     column 5='Left Cingulum Cingulate'
    %     column 6='Right Cingulum Cingulate'
    %     column 7='Left Cingulum Hippocampus'
    %     column 8='Right Cingulum Hippocampus'
    %     column 9='Callosum Forceps Major'
    %     column 10='Callosum Forceps Minor'
    %     column 11='Left IFOF'
    %     column 12='Right IFOF'
    %     column 13='Left ILF'
    %     column 14='Right ILF'
    %     column 15='Left SLF'
    %     column 16='Right SLF'
    %     column 17='Left Uncinate'
    %     column 18='Right Uncinate'
    %     column 19='Left Arcuate'
    %     column 20='Right Arcuate'
    
    
    %%% last column to show the subjet ID
    allsubject.meanVOLUME(k,21)= str2num(name_subj{k});
    allsubject.meanFA(k,21)= str2num(name_subj{k});
    allsubject.meanMD(k,21)= str2num(name_subj{k});
    allsubject.meanRD(k,21)= str2num(name_subj{k});
    allsubject.meanAD(k,21)= str2num(name_subj{k});
    allsubject.streamline(k,21)= str2num(name_subj{k});
    
    % Switch to next subject
    %=======================================
    k   = k + 1;
    
end  %%% end of main loop


cd(dir_base)

save('mean_tract_properties_across_subjects.mat', 'allsubject');

xlswrite('major_fiber_meanVOLUME.xlsx',allsubject.meanVOLUME);
xlswrite('major_fiber_meanFA.xlsx',allsubject.meanFA);
xlswrite('major_fiber_meanMD.xlsx',allsubject.meanMD);
xlswrite('major_fiber_meanRD.xlsx',allsubject.meanRD);
xlswrite('major_fiber_meanAD.xlsx',allsubject.meanAD);
xlswrite('major_fiber_streamline.xlsx',allsubject.streamline);

