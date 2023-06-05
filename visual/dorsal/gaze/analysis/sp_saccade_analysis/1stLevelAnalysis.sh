#!/bin/bash

# estimates 1st level for all files that have interals in gaze_intervals

subjects=(`ls -d /mnt/scratch/ioannis/studyforrest/sub-*/ | rev | cut -c2-3 | rev`)
#subjects=(01)

preprocDir="/mnt/scratch/ioannis/studyforrest_preprocessed/"
analysisBaseDir="/tmp/sp_sacc_analysis/"

for ((i=0; i<${#subjects[@]}; i++))
do
    subDir="${preprocDir}sub-${subjects[i]}/ses-movie/func"

    intsDir="regressors"


    if [ ! -d "$subDir" ]
    then
        continue
    fi
    
    analysisDir="${analysisBaseDir}sub-${subjects[i]}"
    echo "Processing $subDir. Results are stored in $analysisDir"

    matlab -nodisplay -nodesktop -r "try; Create1stLevelMat('${analysisDir}', '${subDir}', '${intsDir}', '${subjects[i]}'); catch; end; quit;"
done

