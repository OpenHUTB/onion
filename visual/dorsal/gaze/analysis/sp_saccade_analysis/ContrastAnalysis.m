% ContratAnalysis.m
%
% This function creates the contrasts of interest. It uses the SPM.mat file
% within the provided directory
%
% input:
%   directory   - directory to search for SPM.mat file

function ContrastAnalysis(directory)
    % create batch file
    matlabbatch = [];
    matlabbatch{1}.spm.stats.con.spmmat = cellstr([directory '/SPM.mat']);

    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'SP > sacc';
    contrast = zeros(8,18);
    contrast(:,4) = 1;
    contrast(:,10) = -1;
    contrast = constrast';
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.convec = contrast(:);
    % equivalent to
%    matlabbatch{1}.spm.stats.con.consess{1}.tcon.convec = [0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ...
%                                                    0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 2
%                                                    0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 3
%                                                    0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 4
%                                                    0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 5
%                                                    0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 6
%                                                    0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 7
%                                                    0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 8
%                                                    ];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'sacc > SP';
    contrast = zeros(8,18);
    contrast(:,4) = -1;
    contrast(:,10) = 1;
    contrast = constrast';
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'SP > baseline';
    contrast = zeros(8,18);
    contrast(:,4) = 1;
    contrast = constrast';
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'sacc > baseline';
    contrast = zeros(8,18);
    contrast(:,10) = 1;
    contrast = constrast';
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

    % execute job
    spm_jobman('run',matlabbatch);
end
