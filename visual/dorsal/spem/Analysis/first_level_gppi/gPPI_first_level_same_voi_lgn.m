%-----------------------------------------------------------------------
% Script by Rebekka Schröder (rebekka.schroeder@uni-bonn.de) to analyse
% data for smooth pursuit connectivity study

% first-level gPPI analysis for left and right LGN seed regions
%-----------------------------------------------------------------------

project_path = 'your_project_path'; % please customize to your project path
cd(project_path);

% project path should contain study folders that contain the individual
% first-level subject folders that contain a folder called "SPEM" that
% contain a folder called "nonparametric" with the first-level data
% e. g. a first-level path could look like this:
% C:\Data\study_1\subject_33\SPEM\nonparametric


study_dir = list_vp_names(project_path);

% same start voxel for every participant
% -24, -26, -4;22, -26, 0

for j = 1:numel(study_dir)

study_path = [project_path filesep study_dir{j}];
subjects = list_vp_names(study_path);

voxel_names = { 'LGNl4' 'LGNr4'};
voxel_mni = [-24, -26, -4;22, -26, 0];

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