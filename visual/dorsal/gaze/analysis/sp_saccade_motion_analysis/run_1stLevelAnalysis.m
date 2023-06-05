% replace 1stLevelAnalysis.sh
% 运行一阶段分析（个体层次）
subjects = dir('/data2/whd/workspace/data/neuro/software/ds000113');  % 12
preprocDir="/data2/whd/workspace/data/neuro/software/ds000113/";
analysisBaseDir="/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/";
% if exist(analysisBaseDir, 'dir')
%     rmdir(analysisBaseDir, 's');
% end
% mkdir(analysisBaseDir);

% 对每个受试者的目录分别进行处理
for i = 1 : length(subjects)
    if strcmpi(subjects(i).name, '.') == 1 || strcmpi(subjects(i).name, '..') == 1
        continue;
    end
    subDir = fullfile(preprocDir, subjects(i).name, 'ses-movie', 'func');
    initsDir = 'regressors';  % 回归变量（平滑跟踪、扫视）目录（由ComputeRegressorAll.m计算获得，每一行：开始时间、持续时间、sp值）
    analysisDir = fullfile(analysisBaseDir, subjects(i).name);
    sub_name = subjects(i).name;
    Create1stLevelMat(char(analysisDir), char(subDir), initsDir, sub_name(5:6));
end
% Create1stLevelMat('/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/sub-01', ...
%  '/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func', ...
%   'regressors', '01');
