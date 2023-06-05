#!/bin/sh
#PBS -N face_REST_STEP1_L1rest_1st_28SS
#PBS -q large
#PBS -l nodes=2:ppn=16
#PBS -l walltime=48:00:00
#PBS

cd $PBS_O_WORKDIR

module load fsl/5.0.10
fslinit

rm -f cmd_${PBS_JOBID}.txt
touch cmd_${PBS_JOBID}.txt

for task in REST1 REST2; do
  for subj in `cat sublist_1st_28ss`; do
    for RUN in LR RL; do
      for H in R L; do

        SCRIPTNAME=L1rest.txt
        echo bash $SCRIPTNAME $task $RUN $subj $H >> cmd_${PBS_JOBID}.txt
        
      done
    done
  done
done

torque-launch -p chkpt_${PBS_JOBID}.txt cmd_${PBS_JOBID}.txt
