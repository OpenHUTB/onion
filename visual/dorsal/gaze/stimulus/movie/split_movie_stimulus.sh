
if false;then
## get whole Stimuls from film 'Forrest Gamp'
# forrest_gump.mkv -> forrest_gump_bluray_orig.mkv(.00 -> 1:58:06)
melt forrest_gump.mkv force_fps=25.000 in="00:00:00:00" out="00:21:32:12" \
     forrest_gump.mkv force_fps=25.000 in="00:24:13:24" out="00:38:31:23" \
     forrest_gump.mkv force_fps=25.000 in="00:38:58:20" out="00:57:19:22" \
     forrest_gump.mkv force_fps=25.000 in="00:59:31:20" out="01:18:14:00" \
     forrest_gump.mkv force_fps=25.000 in="01:20:24:16" out="01:34:18:06" \
     forrest_gump.mkv force_fps=25.000 in="01:37:14:19" out="01:41:30:19" \
     forrest_gump.mkv force_fps=25.000 in="01:42:49:19" out="02:09:51:17" \
  -consumer avformat:forrest_gump_bluray_orig.mkv acodec=libmp3lame vcodec=libx264
fi


## Split whole Stimuls into 8 segments - one for each fMRI recording run
melt forrest_gump_bluray_orig.mkv in="00:00:00.00" out="00:14:52.19" \
  -consumer avformat:fg_av_ger_seg0.mkv acodec=libmp3lame \
& \
melt forrest_gump_bluray_orig.mkv in="00:14:52.19" out="00:29:21:01" \
  -consumer avformat:fg_av_ger_seg1.mkv acodec=libmp3lame \
& \
melt forrest_gump_bluray_orig.mkv in="00:29:21:01" out="00:43:40:51" \
  -consumer avformat:fg_av_ger_seg2.mkv acodec=libmp3lame \
& \
melt forrest_gump_bluray_orig.mkv in="00:43:40:51" out="00:59:40:07" \
  -consumer avformat:fg_av_ger_seg3.mkv acodec=libmp3lame \
& \
melt forrest_gump_bluray_orig.mkv in="00:59:40:07" out="01:14:49:41" \
  -consumer avformat:fg_av_ger_seg4.mkv acodec=libmp3lame \
& \
melt forrest_gump_bluray_orig.mkv in="01:14:49:41" out="01:29:01:37" \
  -consumer avformat:fg_av_ger_seg5.mkv acodec=libmp3lame \
& \
melt forrest_gump_bluray_orig.mkv in="01:29:01:37" out="01:46:50:35" \
  -consumer avformat:fg_av_ger_seg6.mkv acodec=libmp3lame \
& \
melt forrest_gump_bluray_orig.mkv in="01:46:50:35" out="01:57:58:29" \
  -consumer avformat:fg_av_ger_seg7.mkv acodec=libmp3lame
