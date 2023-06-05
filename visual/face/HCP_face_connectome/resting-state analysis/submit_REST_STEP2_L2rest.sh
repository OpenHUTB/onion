#!/bin/sh
#PBS -N face_REST_STEP2_L2rest_1st_28SS
#PBS -q normal
#PBS -l nodes=4:ppn=28
#PBS -l walltime=48:00:00
#PBS

cd $PBS_O_WORKDIR

module load fsl/5.0.10
fslinit

rm -f cmd_${PBS_JOBID}.txt
touch cmd_${PBS_JOBID}.txt


for task in REST1 REST2; do
  for subj in `cat sublist_1st_28ss`; do
    for H in R L; do

      SCRIPTNAME=L2rest.txt
      echo bash $SCRIPTNAME $task $subj $H >> cmd_${PBS_JOBID}.txt

    done
  done
done

torque-launch -p chkpt_${PBS_JOBID}.txt cmd_${PBS_JOBID}.txt

# To save disk space, we need to delete the L1 results 
for subj in `cat sublist_1st_28ss`; do
	rm -rf /home/yw/work/face_PPI/fsl/${subj}/MNINonLinear/Results/rfMRI_REST1_LR
	rm -rf /home/yw/work/face_PPI/fsl/${subj}/MNINonLinear/Results/rfMRI_REST1_RL
	rm -rf /home/yw/work/face_PPI/fsl/${subj}/MNINonLinear/Results/rfMRI_REST2_LR
	rm -rf /home/yw/work/face_PPI/fsl/${subj}/MNINonLinear/Results/rfMRI_REST2_RL
done