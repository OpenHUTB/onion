
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg0.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg0_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv \
& \
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg1.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg1_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-2_recording-eyegaze_physio.tsv \
& \
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg2.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg2_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-3_recording-eyegaze_physio.tsv \
& \
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg3.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg3_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-4_recording-eyegaze_physio.tsv \
& \
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg4.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg4_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-5_recording-eyegaze_physio.tsv \
& \
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg5.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg5_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-6_recording-eyegaze_physio.tsv \
&
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg6.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg6_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-7_recording-eyegaze_physio.tsv \
&
/home/d/anaconda2/envs/hart/bin/python /data2/whd/workspace/sot/hart/gaze/overlay_gaze_on_video \
  /data2/whd/workspace/data/neuro/movie/fg_av_ger_seg7.mkv  \
  /data2/whd/workspace/sot/hart/result/gaze/fg_av_ger_seg7_et.mkv  \
  /data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-7_recording-eyegaze_physio.tsv

# kill all processer
# ps aux | grep overlay_gaze_on_video | awk '{print $2}' | xargs kill -s 9