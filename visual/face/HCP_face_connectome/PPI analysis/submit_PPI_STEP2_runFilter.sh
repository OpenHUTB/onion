#!/bin/bash
#PBS -N face_PPI_STEP2_runFilter_2nd_45ss
#PBS -q normal
#PBS -l nodes=4:ppn=28
#PBS -l walltime=12:00:00
#PBS

cd $PBS_O_WORKDIR

source myenv/bin/activate

module load fsl/5.0.10
source ${FSLDIR}/etc/fslconf/fsl.sh

rm -f cmd_${PBS_JOBID}.txt
touch cmd_${PBS_JOBID}.txt

# This script will apply a 200 s highpass filter
# This is done to ensure the input timecourses for the PPI analyses have been filtered
# Simply checking the "apply highpass filter" box in Feat does not apply the filter before calculating the interaction

for subj in `cat sublist_2nd_45ss`; do
	for RUN in LR RL; do

		SCRIPTNAME=runFilter.txt
		echo bash $SCRIPTNAME $RUN $subj >> cmd_${PBS_JOBID}.txt

	done
done

torque-launch -p chkpt_${PBS_JOBID}.txt cmd_${PBS_JOBID}.txt
