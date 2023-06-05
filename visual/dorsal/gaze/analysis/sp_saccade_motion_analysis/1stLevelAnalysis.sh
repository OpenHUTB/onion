#!/bin/bash

# estimates 1st level for all files that have interals in gaze_intervals

# rev - reverse lines characterwise
# cut - remove sections from each line of files
#       -c2-3, select only these characters
subjects=(`ls -d /data2/whd/workspace/data/neuro/software/ds000113/sub-*/ | rev | cut -c2-3 | rev`)
#subjects=(01)

# preprocess fMRI data direcotry
preprocDir="/data2/whd/workspace/data/neuro/software/ds000113/"
# output directory
analysisBaseDir="/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/"

for ((i=0; i<${#subjects[@]}; i++))
do
    subDir="${preprocDir}sub-${subjects[i]}/ses-movie/func"

    intsDir="regressors"


    if [ ! -d "$subDir" ]
    then
        continue
    fi
    
    analysisDir="${analysisBaseDir}sub-${subjects[i]}"
    #echo "Processing $subDir. Results are stored in $analysisDir"

    # (outputDir, scansDir, intervalsDir, subjId)
    echo ${analysisDir} ${subDir} ${intsDir} ${subjects[i]}
    #matlab -nodisplay -nodesktop -r "try; Create1stLevelMat('${analysisDir}', '${subDir}', '${intsDir}', '${subjects[i]}'); catch; end; quit;"
done
