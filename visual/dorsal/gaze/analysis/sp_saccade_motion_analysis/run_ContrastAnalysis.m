%% 不同眼动类型之间对比分析
run('../init_env.m');

analysisBaseDir = '/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/';

subjects = dir(analysisBaseDir);

for i = 1 : length(subjects)
    if strcmpi(subjects(i).name, '.') == 1 || strcmpi(subjects(i).name, '..') == 1
        continue;
    end
    sub_dir = fullfile(subjects(i).folder, subjects(i).name);
    ContrastAnalysis(sub_dir);
end
