
% add path for "xjview"
run('../gaze/analysis/init_env.m');

if isunix
    xjview('/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/sub-01/spmT_0001.nii');
else
    xjview('Z:/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/sub-01/spmT_0001.nii');
end

