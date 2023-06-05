


## Introduction
mk_movie_stimulus.sh
process raw film.

extract_all_file.m
extract all .gz file.

overlay_gaze_on_video (display_gaze_fMRI.m)
display film gaze in the left and corresponding fMRI data in the right.

## Q&A
install LADSPA audio processing plugins
sudo apt-get install libfftw3-dev
https://github.com/swh/ladspa

./pip install pymvpa2
sudo apt-get install swig

install cv2
cd /home/d/anaconda2/envs/hart/bin
./pip install scikit-build

3.1.0 read() None
./pip install opencv-python==4.2.0.32

 
https://blog.csdn.net/majinlei121/article/details/78192284
wget http://www.ffmpeg.org/releases/ffmpeg-3.1.tar.gz
wget -O opencv.zip https://github.com/Itseez/opencv/archive/2.4.9.zip


d@d:~/anaconda2/envs/hart/bin$ ./pip  uninstall opencv-python
Uninstalling opencv-python-3.1.0.0:
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  cv2/__init__.py
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  cv2/__init__.pyc
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  cv2/cv2.so
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  opencv_python-3.1.0.0.dist-info/DESCRIPTION.rst
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  opencv_python-3.1.0.0.dist-info/INSTALLER
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  opencv_python-3.1.0.0.dist-info/METADATA
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  opencv_python-3.1.0.0.dist-info/RECORD
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  opencv_python-3.1.0.0.dist-info/WHEEL
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  opencv_python-3.1.0.0.dist-info/metadata.json
  /home/d/anaconda2/envs/hart/lib/python2.7/site-packages/  opencv_python-3.1.0.0.dist-info/top_level.txt


'xlocale.h' file not found when compile mlt-0.8.0
ln -s /usr/include/locale.h /usr/include/xlocale.h

alsa/asoundlib.h: No such file or directory when compile mlt-6.24.0
sudo apt-get install libasound2-dev

pulse/error.h: No such file or directory  when compile mlt-6.24.0
sudo apt-get install libpulse-dev 
