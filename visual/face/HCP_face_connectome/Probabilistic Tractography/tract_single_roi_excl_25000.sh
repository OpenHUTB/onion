for s in L_AMG L_ATL L_FFA L_IFG L_OFA L_OFC L_PCC L_PPA L_pSTS L_V1 R_AMG R_ATL R_FFA R_IFG R_OFA R_OFC R_PCC R_PPA R_pSTS R_V1
do
  mkdir -p /path/tract_single_roi_excl_25000/${s}  
  for n in 263436 627549
  do
      probtrackx2 -x /path/spheres/${n}/${s}.nii  -l --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P 25000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=/data/projects/fmri/${n}/MNINonLinear/xfms/standard2acpc_dc.nii.gz --invxfm=/data/projects/fmri/${n}/MNINonLinear/xfms/acpc_dc2standard.nii.gz --avoid=/path/cerebellum_mask.nii.gz --forcedir --opd -s /gpfs/projects/fmri/bedpostX_gpu_g_rician/${n}.bedpostX/merged -m /gpfs/projects/fmri/bedpostX_gpu_g_rician/${n}.bedpostX/nodif_brain_mask  --dir=/path/tract_single_roi_excl_25000/${s}/${n}
  done
done