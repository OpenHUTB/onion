#!/bin/sh
#PBS -N face_PPI_STEP4_L2ppi_small_size_2nd_45ss
#PBS -q normal
#PBS -l nodes=4:ppn=28
#PBS -l walltime=12:00:00
#PBS

cd $PBS_O_WORKDIR

module load fsl/5.0.10
fslinit

rm -f cmd_${PBS_JOBID}.txt
touch cmd_${PBS_JOBID}.txt


for subj in `cat sublist_2nd_45ss`; do
  for H in R L; do
    for PPItype in full partial; do
      for PPIseed in V1 OFA FFA ATL pSTS IFG AMG OFC PCC; do

        SCRIPTNAME=L2ppi.txt
        echo bash $SCRIPTNAME $subj $H $PPIseed $PPItype >> cmd_${PBS_JOBID}.txt

      done
    done
  done
done

torque-launch -p chkpt_${PBS_JOBID}.txt cmd_${PBS_JOBID}.txt

# To save disk space, we need to delete the L1 results in tfMRI_WM_LR and tfMRI_WM_RL (each ~9GB)
for subj in `cat sublist_2nd_45ss`; do
	rm -rf /home/yw/work/face_PPI/fsl/${subj}/MNINonLinear/Results/tfMRI_WM_LR
	rm -rf /home/yw/work/face_PPI/fsl/${subj}/MNINonLinear/Results/tfMRI_WM_RL
done