
%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% first-level gPPI analysis for left and right V5 seed regions
%-----------------------------------------------------------------------

project_path = 'your_project_path'; % please customize to your project path

% project path should contain study folders that contain the individual
% first-level subject folders that contain a folder called "SPEM" that
% contain a folder called "nonparametric" with the first-level data
% e. g. a first-level path could look like this:
% C:\Data\study_1\subject_33\SPEM\nonparametric


cd(project_path);

study_dir = list_vp_names(project_path);

% analysis uses individual voxels for each participant: 

% fo find these individual voxels omnibus-F-contrast for each
% participant was opened (uncorrected, thresholded at .001) 
% V5 coordinates (-44, -76, 2; 43, -73, 5; from table 2 Wilms; Wilms, Marcus; Eickhoff, 
% Simon B.; Specht, Karsten; Amunts, Katrin; Shah, Nadim J. et al. (2005): Human V5/MT+: 
% comparison of functional and cytoarchitectonic data. In: Anatomy and embryology 210 (5-6), 
% S. 485–495. DOI: 10.1007/s00429-005-0064-y&pii=64.) were entered 
% --> go to nearest local maximum was selected in the GUI
% --> nearest maximum and distance were saved along with subject id to v5_coordinates.txt in the project path



for j = 1:numel(study_dir)

study_path = [project_path filesep study_dir{j}];
subjects = list_vp_names(study_path);

voxel_names = { 'V5lin' 'V5rin'};

% Add the path to the gPPI toolbox. 

addpath 'C:/spm';
spms(8);
spmpath=fileparts(which('spm.m'));
addpath([spmpath filesep 'toolbox']);
addpath([spmpath filesep 'toolbox' filesep 'PPPI']);
spm('defaults','fmri')

for k = 1:numel(subjects)
    
    sb_name = subjects{k};
        disp(sb_name)
        disp('analyzing')
    
workdir= [study_path filesep subjects{k}];
cd(workdir);

first_level_dir = [workdir filesep 'SPEM' filesep 'nonparametric'];



cd(project_path);

A = readtable('v5_coordinates.txt');
A.participant(35:45) = strcat(A.participant(35:45), '_Placebo');
where = find(strcmp(sb_name,A.participant));
voxel_mni = [table2array(A(where,3:5)); table2array(A(where,7:9))];

cd(first_level_dir);

% create sphere around voxel of interest

for i = 1:numel(voxel_names)

cd(first_level_dir);
create_sphere_image('SPM.mat',voxel_mni(i,:),voxel_names(i),4);

% Set up the control param structure P which determines the run
cd(workdir);

P.subject=subjects{k};
P.directory=[workdir filesep 'SPEM' filesep 'nonparametric'];
P.VOI= [workdir filesep 'SPEM' filesep 'nonparametric' filesep strcat(voxel_names{i}, '_mask.nii')];
P.Region=voxel_names{i};
P.analysis='psy';
P.method='cond';
P.Estimate=1;
P.extract='eig';
P.Tasks={'1' 'SPEM .2' 'SPEM .4'}; 
P.equalroi=0;
P.FLmask=1;
P.CompContrasts=1;
P.outdir = [workdir filesep 'SPEM' filesep 'gPPI'];

  try
    PPPI(P);
  catch ME
    fprintf('error in ppi');
  end



end
end
end