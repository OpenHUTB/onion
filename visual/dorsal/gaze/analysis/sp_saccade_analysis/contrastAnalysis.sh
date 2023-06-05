#!/bin/bash
#
# calls ContrastAnalysis.m for each directory in results

analysisBaseDir="/tmp/sp_sacc_analysis/"
for dir in `ls -d ${analysisBaseDir}sub-*/`
do
    echo "Processing $dir"
    matlab -nodisplay -nodesktop -r "try; ContrastAnalysis('$dir'); catch; end; quit;"
done                            
