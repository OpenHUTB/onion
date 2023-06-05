# Connectivity Project

These scripts will help conduct functional and effective connectivity analyses (PPI). We are mainly focusing on the WM task and isolating the face contrasts.

## Resting State Analysis
### Level 1 Analyses
1. Update paths in L1rest.sh
1. Test on one subject/run: `bash L1rest.sh REST1 LR 100307 L`
1. Update submission script (submit_L1rest.sh) to work with cluster

### Level 2 Analyses
1. Update paths in L2rest.sh
1. Test on one subject/run: `bash L2rest.sh 100307 L`
1. Update submission script (submit_L2rest.sh) to work with cluster.


## PPI Analyses
For this set of analyses, we have a few more things to consider in terms of preprocessing. Note that you will likely have to install a couple of Python packages to get ICA-AROMA to work on your system.

### Preprocessing

1. Update paths in runAROMA.sh (this smoothes the data and removes motion)
1. Update paths in runFilter.sh (this applies a hp filter since AROMA uses unfiltered data)
1. Test on one subject/run: `bash runAROMA.sh WM LR 100307`
1. Test on one subject/run: `bash runFilter.sh LR 100307`
1. Update submission scripts (submit_runAROMA.sh and submit_runFilter.sh) to work with cluster.
1. First run `submit_runAROMA.sh` on all data. Then run `submit_runFilter.sh` on all of the outputs of runAROMA.sh. (order here is essential)

### Level 1 Analyses

Now that the data have been pre-processed in a way that resembles the REST data, we can run our PPI models. For that, we have two basic PPI models. The first one is "full" and includes all of the ROI timecourses (from one hemisphere); and the second one is "partial" and includes only the seed region of interest. In my view, the "full" model is what we want since it is closer to the REST analyses because it controls for responses in the other regions. Here's how to run it:
1. Update paths in L1ppi.sh
1. Test on one subject/run: `bash L1ppi.sh LR 100307 L OFC full`
1. Update submission script (submit_L1ppi.sh) to work with cluster.

### Level 2 Analyses
1. Update paths in L2ppi.sh
1. Test on one subject/run: `bash L2ppi.sh 100307 L OFC full`
1. Update submission script (submit_L2ppi.sh) to work with cluster.
