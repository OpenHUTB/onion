%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% second level analysis of gppi for right LGN as a seed region
%-----------------------------------------------------------------------


% for the script so run smoothly first level PPI names should be saved along the following hierarchy:
% data_path    <- directory where first level data is saved, contains folders
%                 for each study: 
% study_dir    <- contain one folder for each participant with a 
%                 unique name ('subject_name', e.g. 'subject_33'),
%                 contains folder called:
% SPEM         <- contains folder called:
% gPPI         <- contains separate folders for each seed region: 
% voi_dir      <- named ("PPI_" name of seed region), e. g. "PPI_V5lin",
%                 contains first-level images that are named like this: 
% image_name  <-  (contrast name "_" subject_name ".img")
%                 for each participant, 8 contrasts exist, named like this: 
%                   con_PPI_none_minus_SPEM .2
%                   con_PPI_none_minus_SPEM .4
%                   con_PPI_none_minus_SPEM .2_SPEM .4
%                   con_PPI_SPEM .2_minus_SPEM .4
%                   con_PPI_SPEM .2_minus_none
%                   con_PPI_SPEM .2_SPEM .4_minus_none
%                   con_PPI_SPEM .4_minus_none
%                   con_PPI_SPEM .4_minus_SPEM .2

% example of a first-level path:
% C:\data\study_1\subject_33\SPEM\gPPI\PPI_V5lin\con_PPI_none_minus_SPEM.2_subject_33.img


% your project path should contain folders for each seed region, e. g.
% "V5lin_without_critical" that contain the following subfolders (one for each contrast): 
%                   con_PPI_none_minus_SPEM .2
%                   con_PPI_none_minus_SPEM .4
%                   con_PPI_none_minus_SPEM .2_SPEM .4
%                   con_PPI_SPEM .2_minus_SPEM .4
%                   con_PPI_SPEM .2_minus_none
%                   con_PPI_SPEM .2_SPEM .4_minus_none
%                   con_PPI_SPEM .4_minus_none
%                   con_PPI_SPEM .4_minus_SPEM .2
%

% example: 
% C:\analysis\V5lin_without_critical\con_PPI_SPEM .4_minus_SPEM .2

%%
clear all

project_path = 'your_project_path'; % please customize to your project path, where results should be saved
cd(project_path);

voi_dir = {'LGNr4'};   % please customize
num_participants = 57;  % please customize (number of participants)
    
exclusion_participants = {'exclusion_participants'}; % please customize 
% (character vector of participants (their unique name) that should be excluded from second level analysis)
        

% find scans

data_path = 'your_data_path'; % please customize to your data path (where first-level data is saved)
cd(data_path);



for j = 1:numel(voi_dir)

voi_path = [project_path filesep strcat(voi_dir{j}, '_without_critical')];
contrasts = list_vp_names(voi_path);             
              
for k = 1:numel(contrasts)


contrast_dir = [voi_path filesep contrasts{k}];
cd(contrast_dir);

scan_names = ['PPI_' voi_dir{j} filesep contrasts{k} '*.img'];
    

cd(data_path);
study_dirs = list_vp_names(data_path); 


scan_names_all = {};

for y=1:numel(study_dirs)
   
subject_names = list_vp_names(strcat(data_path, '\', study_dirs{y}));
    


scan_names = strcat('your_data_path', study_dirs{y}, '\', subject_names, ...                        % customize 'your_data_path'
                    '\SPEM\gPPI\PPI_', voi_dir{j},'\', contrasts{k}, '_', subject_names, '.img')';

                
scan_names_all = [scan_names_all; scan_names];

end




%exclude participants

for i=1:numel(exclusion_participants)
    
index = strfind(scan_names_all, exclusion_participants{i});
non_empties = ~cellfun('isempty',index);
scan_names_all(non_empties) = [];
end



matlabbatch{1}.spm.stats.factorial_design.dir = {contrast_dir};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = scan_names_all;
%
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% run batch
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, '');

% save batch

cd(char(matlabbatch{1}.spm.stats.factorial_design.dir))
batch_name = [contrasts{k} '.mat'];
save (batch_name, 'matlabbatch');
save ('scans', 'scan_names_all');

end
end
