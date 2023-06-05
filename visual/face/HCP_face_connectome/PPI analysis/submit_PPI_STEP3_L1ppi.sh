#!/bin/sh
#PBS -N face_PPI_STEP3_L1ppi_2nd_45ss
#PBS -q normal
#PBS -l nodes=12:ppn=28
#PBS -l walltime=48:00:00
#PBS

cd $PBS_O_WORKDIR

module load fsl/5.0.10
fslinit

rm -f cmd_${PBS_JOBID}.txt
touch cmd_${PBS_JOBID}.txt

for subj in `cat sublist_2nd_45ss`; do
  for RUN in LR RL; do
    for H in R L; do
      for PPItype in full partial; do
        for PPIseed in V1 OFA FFA ATL pSTS IFG AMG OFC PCC; do

          SCRIPTNAME=L1ppi.txt
          echo bash $SCRIPTNAME $RUN $subj $H $PPIseed $PPItype  >> cmd_${PBS_JOBID}.txt

        done
      done
    done
  done
done

torque-launch -p chkpt_${PBS_JOBID}.txt cmd_${PBS_JOBID}.txt
