#!/bin/bash
#
# calls ContrastAnalysis.m for each directory in results

analysisBaseDir="/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/"
for dir in `ls -d ${analysisBaseDir}sub-*/`
do
    echo "Processing $dir"
    matlab -nodisplay -nodesktop -r "try; ContrastAnalysis('$dir'); catch; end; quit;"
done                            
