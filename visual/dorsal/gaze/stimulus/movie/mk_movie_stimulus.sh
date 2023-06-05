##end credits music (剧终感谢音乐)
# melt -profile hdv_720_25p -video-track gray_screen.png out=9398 -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_5s.flac forrest_gump_ger_stereo.flac in=195024 out=204422 -mix 125 -mixer mix:-1 -consumer avformat:fg_av_ger_seg99.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


##segment: 0
##frames: 22550.0
##TRs: 451.0 (Time of Repetition)
##Duration: 902.0 s
# multitrack command line help (https://www.mltframework.org/docs/melt/)
## -profile:        Set the processing settings
## -video-track:    Add a video-only track
## -attach-track:   filter[:arg] [name=value]* Attach a filter(frame modifiers) to a track
## -audio-track: 		Add an audio-only track
## -transition id[:arg] [name=value]*        Add a transition
## -mix length:      Add a mix between the last two cuts
## -mixer transition:  Add a transition to the mix
# out=25: the first second contains a watermark (in= out= : split video with time)
# .flac - Free Lossless Audio Codec
# Linux Audio Developer's Simple Plugin API (LADSPA)
# acodec: audio decoder
# vcodec: video decoder


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

#melt forrest_gump_bluray_orig.mkv in=0 out=21900 \
#  -consumer avformat:fg_av_ger_seg2.mkv acodec=libmp3lame

#melt forrest_gump_bluray_orig.mkv in=0 out=24400 \
#  -consumer avformat:fg_av_ger_seg3.mkv acodec=libmp3lame


#melt forrest_gump_bluray_orig.mkv in="00:00:00:00" out="00:15:02:00" \
#  -consumer avformat:fg_av_ger_seg0.mkv acodec=libmp3lame


melt -video-track \
  0.mkv out=25 fixation.png out="00:09:15:18" -mix 25 -mixer luma \
  fixation.png out="00:14:46:03" gray.png out=100 -mix 250 mixer luma \
  -consumer avformat:fg_av_ger_seg0.mkv acodec=libmp3lame


if false;then
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 \
 forrest_gump_bluray_orig.mkv force_fps=25.000 in=35 out=22585 \
 -mix 25 -mixer luma gray_screen_hdv720.png out=200 \
 -mix 100 -mixer luma \
 -video-track \
 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 \
 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 \
 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=2550 \
 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite \
 -audio-track -attach-track ladspa.1197 0=-70 1=-10 \
 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 \
 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 \
 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac \
 in=0 out=22550 \
 -mix 25 -mixer mix:-1 silence_5s.flac \
 -mix 100 -mixer mix:-1 \
 -consumer avformat:fg_av_ger_seg0.mkv \
 f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k
fi

# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=35 out=22585 -mix 25 -mixer luma gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=2550 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=0 out=22550 -mix 25 -mixer mix:-1 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg0.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

# render movie segments
#melt -video-track \
#gray.png out=25 fixation.png out=14000 -mix 25 -mixer luma \
#fixation.png out=8550 gray.png out=100 -mix 250 -mixer luma \
#-audio-track AUDIOFILTER \
#silence_1s.flac ad_ger_stereo.flac in=0 out=22550 -mix 25 -mixer mix:-1 \
#silence_5s.flac -mix 100 -mixer mix:-1 \
#-consumer avformat:fg_ad_seg0.mkv OUTPUTSPEC


##segment: 1
##frames: 22050.0
##TRs: 441.0
##Duration: 882.0
melt -profile hdv_720_25p \
  -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 \
  forrest_gump_bluray_orig.mkv force_fps=25.000 in=22185 out=32348 \
  -mix 25 -mixer luma \
  forrest_gump_bluray_orig.mkv force_fps=25.000 in=36385 out=48273 \
  gray_screen_hdv720.png out=200 \
  -mix 100 -mixer luma \
  -video-track  \
  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 \
  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 \
  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=2050 \
  -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite \
  -audio-track -attach-track ladspa.1197 0=-70 1=-10 \
  -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 \
  -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 \
  -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=22150 out=32312 \
  -mix 25 -mixer mix:-1 forrest_gump_ger_stereo.flac in=36349 out=48237 silence_5s.flac \
  -mix 100 -mixer mix:-1 \
  -consumer avformat:fg_av_ger_seg1.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k
# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=22185 out=32348 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=36385 out=48273 gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=2050 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=22150 out=32312 -mix 25 -mixer mix:-1 forrest_gump_ger_stereo.flac in=36349 out=48237 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg1.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


##segment: 2
##frames: 21900.0
##TRs: 438.0
##Duration: 876.0
# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=47873 out=57835 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=58507 out=70446 gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=1900 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=47837 out=57799 -mix 25 -mixer mix:-1 forrest_gump_ger_stereo.flac in=58471 out=70409 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg2.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#segment: 3
#frames: 24400.0
#TRs: 488.0
#Duration: 976.0
# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=70046 out=86036 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=89332 out=97742 gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=4400 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=70009 out=85999 -mix 25 -mixer mix:-1 forrest_gump_ger_stereo.flac in=89295 out=97705 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg3.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#segment: 4
#frames: 23100.0
#TRs: 462.0
#Duration: 924.0
# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=97342 out=117391 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=120656 out=123708 gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=3100 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=97305 out=117353 -mix 25 -mixer mix:-1 forrest_gump_ger_stereo.flac in=120618 out=123670 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg4.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#segment: 5
#frames: 21950.0
#TRs: 439.0
#Duration: 878.0
# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=123308 out=141496 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=145908 out=149671 gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=1950 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=123270 out=141457 -mix 25 -mixer mix:-1 forrest_gump_ger_stereo.flac in=145869 out=149632 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg5.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#segment: 6
#frames: 27100.0
#TRs: 542.0
#Duration: 1084.0
# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=149271 out=152304 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=154288 out=178356 gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=7100 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=149232 out=152265 -mix 25 -mixer mix:-1 forrest_gump_ger_stereo.flac in=154249 out=178316 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg6.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#segment: 7
#frames: 16876.0
#TRs: 337.52
#Duration: 675.04
# melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=177956 out=194832 -mix 25 -mixer luma gray_screen_hdv720.png out=200 -mix 100 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=9999 meta.media.width=1280 meta.media.height=720 indicator/.all.png ttl=1 aspect_ratio=1 progressive=1 out=6876 -attach-track affine transition.geometry=1245/690:50x30:100 -transition composite -audio-track -attach-track ladspa.1197 0=-70 1=-10 -attach-track ladspa.2152 0=128 1=502 2=0 3=20 6=3 -attach-track ladspa.2152 0=128 1=502 2=0 3=-20 6=10 -attach-track volume gain=3 silence_1s.flac forrest_gump_ger_stereo.flac in=177916 out=194792 -mix 25 -mixer mix:-1 silence_5s.flac -mix 100 -mixer mix:-1 -consumer avformat:fg_av_ger_seg7.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#entire movie cut: english
# melt -profile hdv_720_25p -video-track forrest_gump_bluray_orig.mkv force_fps=25.000 in=35 out=32348 forrest_gump_bluray_orig.mkv force_fps=25.000 in=36385 out=57835 forrest_gump_bluray_orig.mkv force_fps=25.000 in=58507 out=86036 forrest_gump_bluray_orig.mkv force_fps=25.000 in=89332 out=117391 forrest_gump_bluray_orig.mkv force_fps=25.000 in=120656 out=141496 forrest_gump_bluray_orig.mkv force_fps=25.000 in=145908 out=152304 forrest_gump_bluray_orig.mkv force_fps=25.000 in=154288 out=194832 -audio-track  forrest_gump_eng_stereo.flac in=0 out=32312 forrest_gump_eng_stereo.flac in=36349 out=57799 forrest_gump_eng_stereo.flac in=58471 out=85999 forrest_gump_eng_stereo.flac in=89295 out=117353 forrest_gump_eng_stereo.flac in=120618 out=141457 forrest_gump_eng_stereo.flac in=145869 out=152265 forrest_gump_eng_stereo.flac in=154249 out=194792 -consumer avformat:forrestgump_researchcut_eng.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#entire movie cut: german
# melt -profile hdv_720_25p -video-track forrest_gump_bluray_orig.mkv force_fps=25.000 in=35 out=32348 forrest_gump_bluray_orig.mkv force_fps=25.000 in=36385 out=57835 forrest_gump_bluray_orig.mkv force_fps=25.000 in=58507 out=86036 forrest_gump_bluray_orig.mkv force_fps=25.000 in=89332 out=117391 forrest_gump_bluray_orig.mkv force_fps=25.000 in=120656 out=141496 forrest_gump_bluray_orig.mkv force_fps=25.000 in=145908 out=152304 forrest_gump_bluray_orig.mkv force_fps=25.000 in=154288 out=194832 -audio-track  forrest_gump_ger_stereo.flac in=0 out=32312 forrest_gump_ger_stereo.flac in=36349 out=57799 forrest_gump_ger_stereo.flac in=58471 out=85999 forrest_gump_ger_stereo.flac in=89295 out=117353 forrest_gump_ger_stereo.flac in=120618 out=141457 forrest_gump_ger_stereo.flac in=145869 out=152265 forrest_gump_ger_stereo.flac in=154249 out=194792 -consumer avformat:forrestgump_researchcut_ger.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


#entire movie cut: german AD
# melt -profile hdv_720_25p -video-track forrest_gump_bluray_orig.mkv force_fps=25.000 in=35 out=32348 forrest_gump_bluray_orig.mkv force_fps=25.000 in=36385 out=57835 forrest_gump_bluray_orig.mkv force_fps=25.000 in=58507 out=86036 forrest_gump_bluray_orig.mkv force_fps=25.000 in=89332 out=117391 forrest_gump_bluray_orig.mkv force_fps=25.000 in=120656 out=141496 forrest_gump_bluray_orig.mkv force_fps=25.000 in=145908 out=152304 forrest_gump_bluray_orig.mkv force_fps=25.000 in=154288 out=194832 -audio-track  forrest_gump_ad_ger_stereo.flac in=0 out=32312 forrest_gump_ad_ger_stereo.flac in=36349 out=57799 forrest_gump_ad_ger_stereo.flac in=58471 out=85999 forrest_gump_ad_ger_stereo.flac in=89295 out=117353 forrest_gump_ad_ger_stereo.flac in=120618 out=141457 forrest_gump_ad_ger_stereo.flac in=145869 out=152265 forrest_gump_ad_ger_stereo.flac in=154249 out=194792 -consumer avformat:forrestgump_researchcut_ad_ger.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k


