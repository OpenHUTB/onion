for s in L_AMG_R_AMG L_ATL_R_ATL L_FFA_R_FFA L_IFG_R_IFG L_OFA_R_OFA L_OFC_R_OFC L_PCC_R_PCC L_pSTS_R_pSTS L_V1_R
_V1
do
  mkdir -p /path/tract_interhemisphere/${s}
  for n in 263436 627549
  do
     probtrackx2 --network -x /path/seeds_targets/interhemisphere/${n}/${s}.txt -l --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P 25000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=/data/projects/fmri/${n}/MNINonLinear/xfms/standard2acpc_dc.nii.gz --invxfm=/data/projects/fmri/${n}/MNINonLinear/xfms/acpc_dc2standard.nii.gz --forcedir --opd -s /gpfs/projects/fmri/bedpostX_gpu_g_rician/${n}.bedpostX/merged -m /gpfs/projects/fmri/bedpostX_gpu_g_rician/${n}.bedpostX/nodif_brain_mask  --dir=/path/tract_interhemisphere/${s}/${n}
  done
done