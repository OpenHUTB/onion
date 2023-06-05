Here we provide the code that was used for the analysis of the open source
*studyforrest* data set.

## Install
/home/d/anaconda2/envs/hart/bin/pip install liac-arff numpy scikit-learn python-Levenshtein

## DATA SET

The data set can be downloaded from
[here](https://openneuro.org/datasets/ds000113). After the data set has been
downloaded we can preprocess the functional data and automatically detect eye
movements.

NOTE: the data set structure has changed since it was originally published and
it might be required to change some paths in the preprocessing pipeline.

The preprocessing can be performed with PreprocessRegex Matlab function. ex.
``` 
PreprocessRegex('/tmp/studyforrest/sub-*/') 
```

For the eye movement detection we use the *sp_tool* which is available
[here](https://github.com/MikhailStartsev/sp_tool). Because the *sp_tool* uses
the ARFF file format as input we converted the eye tracking data to the ARFF format with
the *Studyforrest2ArffRegex.m* function. After conversion we called the *sp_tool* with
the provided *default_parameters_studyforrest.conf.json* configuration file as
follows: 
``` 
./run_detection.py --config-file /data2/whd/workspace/sot/hart/gaze/analysis/default_parameters_studyforrest.conf.json
``` 
NOTE: some paths in the configuration file have to be changed in order
to point to the correct directories.

The steps for the conversion of the gaze files and the detection of the eye
movements can be omitted because we provide them in this repository.

Finally in this repository we provide the mean motion of each frame of the
studyforrest videos as computed from the *EpicFlow* algorithm in the *frame_motion* directory.

## fMRI ANALYSIS

For the fMRI analysis we have two directories (*sp_saccade_analysis,
sp_saccade_motion_analysis*) where we have different number of regressors
modelling the respective aspects of the experiment.

NOTE: before progressing further with the analysis you should download some Matlab
utilities that handle ARFF files from 
[here](https://gin.g-node.org/ioannis.agtzidis/matlab_utils) and add them to 
your Matlab search path with the *pathtool* or *addpath* commands.

In order to run the analysis we start by computing the regressors of the 1st
level GLM by running the *ComputeRegressorsAll.m* Matlab function. The call to
this function should work straight out of the box without any tweaking. Then we
call the bash script *1stLevelAnalysis.sh* after changing the used paths in
order to point to the correct directories. Basically the script goes through
the subdirectories that exist for each subject and runs the 1st level analysis
independently for each subject. Then we call the *contrastAnalysis.sh* script,
after changing the paths to point to the correct directories, which computes 
the contrasts of interest. Finally in the 2nd level analysis we computed the
mean group effect by using a simple t-test across all subjects excluding subjects 
05 and 20 due to substantial losses in eye tracking samples.

## OPTIC FLOW EXTRACTION

For optic flow extraction we used the Minors of the Structure Tensor [1] and
the EpicFlow [2] algorithms. The earlier returns the flow in AVI formatted
videos and it does not have a open-source implementation. Therefore we provide
its output in the *minors_flow* folder. The latter is open source and it can be 
downloaded from [here](https://thoth.inrialpes.fr/src/epicflow/). Because EpicFlow
reports 2 float values for each pixel, the resulting files for the studyforrest data
set were disproportionately large (155GB). For this reason we provide the script that
we used (*videoFlow.sh*), which can be run directly from the directory where
the EpicFlow and its dependencies were installed/extracted.

## REFERENCES

[1] Barth, E. (2000). The minors of the structure tensor. In Mustererkennung
2000 (pp. 221-228). Springer, Berlin, Heidelberg.

[2] Revaud, J., Weinzaepfel, P., Harchaoui, Z., & Schmid, C. (2015). Epicflow:
Edge-preserving interpolation of correspondences for optical flow. In
Proceedings of the IEEE conference on computer vision and pattern recognition
(pp. 1164-1172).
