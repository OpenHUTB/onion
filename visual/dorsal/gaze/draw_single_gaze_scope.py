# -*- coding:utf-8 -*-
# 在图片上绘制凝视区域
# python3.6
"""
  示例： 原始视频的路径、存储凝视覆盖视频的路径、tsv凝视数据存储的路径（就像convert_eyelink_to_cont那样处理一样）
  ./draw_single_gaze_scope  D:/data/neuro/movie/fg_av_ger_seg1.mkv  C:/buffer/fg_av_ger_seg1_et.mkv  C:/buffer/sub-01_ses-movie_task-movie_run-2_recording-eyegaze_physio.tsv


该脚本获取凝视时的注意力的范围大小，用圆圈的大小表示，输出到目录: data/seg/{sub_id}/track/gaze_scope.txt
修改自脚本: gaze/overlay_gaze_on_video
Usage:
  ./overlay_gaze_on_video <video_in_path> <video_out_path> <gaze_data_1> [<gaze_data_2> [...]]

  video_in_path - path of original video (readable by opencv library)
  video_out_path - path to store gaze overlayed video
  gaze_data_x - path to gaze data stored as tsv with [x,y,pupil,frame] like convert_eyelink_to_cont does

  OR

  provide a subject code to build a merged eyegaze timeseries across all movie
  segments (with any overlap removed)

Example:
  ./overlay_gaze_on_video fg_av_seg0.mkv fg_av_seg0_et.mkv anondata/sub-01_ses-movie_task-movie_recording-eyegaze_run-1_physio.tsv

  ./overlay_gaze_on_video fg_av_researchcut.mkv fg_et.mkv 01 02 03 ...


Script needs:
  cv2, numpy, pylab, scipy.ndimage

Author:
  Daniel Kottke (daniel.kottke@ovgu.de)
"""


import cv2
import numpy as np
import pylab as plt
import scipy.ndimage as ndi
import sys
import os

# is_debug = True
is_debug = False

### PARAMETERS
# video output options
alpha_video = 1

heatmap = False
show_heatmap = True

# single gaze dots options
alpha_gazes_single = .5

# gaze contour lines options
n_contourlines = 6  # number of contour lines
heatmap_gauss_sigma = 80  # 凝视强度估计 高斯核的标准差。 standard deviation of Gaussian kernel for gaze density estimation
heatmap_scale = 5  # speed-tuning parameter, reduces size of headmap image
alpha_gazes_contourlines = 1

# video position offsets
# gaze coordinates are relative to the actual movie frame, but the video
# files feature a 87px high gray bar at the top
x_offset = 0
# y_offset = 0  # for (1280*546)
y_offset = 87  # for(1280*720)

### declare filenames(DECLARE FILENAMES)
if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)

# 参数1：输入视频路径
video_i_path = str(sys.argv[1])

# 参数2：输出视频路径
video_o_path = str(sys.argv[2])

# 参数3：凝视数据的路径
tsv_path_list = sys.argv[3:]


# opencv等值线函数
def hierarchy_recursion(hierarchy_levels, hierarchy, idx, act_level):
    if idx < 0:
        return hierarchy_levels
    hierarchy_levels[idx] = act_level
    hierarchy_levels = hierarchy_recursion(hierarchy_levels, hierarchy, hierarchy[0, idx, 0], act_level)
    hierarchy_levels = hierarchy_recursion(hierarchy_levels, hierarchy, hierarchy[0, idx, 2], act_level + 1)
    return hierarchy_levels


def get_hierarchy_levels(hierarchy):
    if hierarchy is None:
        return np.array([])
    hierarchy_levels = np.zeros(len(hierarchy[0]), dtype=int) - 1

    for i in range(len(hierarchy_levels)):
        if hierarchy_levels[i] < 0:
            hierarchy_levels = hierarchy_recursion(hierarchy_levels, hierarchy, i, 0)

    return hierarchy_levels


### 读取输入数据
# 打开输入视频
vid = cv2.VideoCapture(video_i_path)

groupcolor = None
if tsv_path_list[0][1:].isdigit():
    groupcolor = [s.startswith('e') for s in tsv_path_list]
    # this is a subject id -> use utility function to load all data
    # pip install pymvpa2 (需要先安装swig）
    from mvpa2.base.hdf5 import h5load, h5save  # Framework for multivariate pattern analysis (MVPA)
    from eyegaze_utils import movie_dataset, preprocess_eyegaze
    data_list = []
    for s in tsv_path_list:
        cachefilename = os.path.join('cache', 'sub-%s_avmovie_eyegaze.hdf5' % s)
        if os.path.exists(cachefilename):
            ds = h5load(cachefilename)
        else:
            if s.startswith('e'):
                ds = movie_dataset(
                    'et%s' % s[1:],
                    preprocess_eyegaze,
                    base_path='../collection/fg_eyegaze',
                    fname_tmpl='sub-%(subj)s/func/sub-%(subj)s_task-movie_run-%(run)i_recording-eyegaze_physio.tsv.gz')
            else:
                ds = movie_dataset(s, preprocess_eyegaze)
            h5save(cachefilename, ds, compression=9)
        data_list.append(ds)
    print(data_list[0])  # 原始代码是python2的
    data_list = [{'x': d.samples[:, 0],
                  'y': d.samples[:, 1],
                  'frame': d.sa.movie_frame} for d in data_list]
else:
    # open data as record array
    data_list = map(lambda fn: np.recfromcsv(fn, names=['x', 'y', 'pup', 'frame'], delimiter='\t'), tsv_path_list)
    data_list = [{'x': d['x'], 'y': d['y'], 'frame': d['frame']} for d in data_list]

### OTHER DATE DECLARATIONS
# create video writer
vid_writer = cv2.VideoWriter()
# virtual bool 	open (const String &filename, int fourcc, double fps, Size frameSize, bool isColor=true)
# vid.get(cv2.CAP_PROP_FPS): 1000? (ref: https://www.imooc.com/article/80856)
vid_writer.open(video_o_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))) )
# vid_writer.open(video_o_path, cv2.VideoWriter_fourcc(*'XVID'), vid.get(cv2.CAP_PROP_FPS), (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

### colormap declaration
def colors_gaze_dots(idx):
    if groupcolor is None:
        return plt.get_cmap('hsv', len(data_list) + 1)(idx)
    else:
        if groupcolor[idx]:
            return (1.0, 0.0, 1.0, 1.0)
        else:
            return (0.0, 1.0, 1.0, 1.0)

colors_contourlines = plt.get_cmap('copper', n_contourlines)
convert2cvcolor = lambda x: np.array(np.array(x[0:3]) * 255, int)


def get_slice(arr, start_idx, crit):
    idx = start_idx
    val = arr[idx]
    l = len(arr)
    start = None
    end = None
    while val <= crit:
        if start is None:
            if val == crit:
                start = idx
        elif val != crit:
            break
        idx += 1
        if idx >= l:
            break
        val = arr[idx]
    end = idx
    if start is None:
        return slice(start_idx, start_idx)
    else:
        return slice(start, end)

start_idx = [0] * len(data_list)

start_frame = os.environ.get('FG_OVERLAY_STARTFRAME', None)
stop_frame = os.environ.get('FG_OVERLAY_STOPFRAME', None)
if not start_frame is None:
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(start_frame))

### loop through all images
while 1:
    frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
    if not stop_frame is None and frame >= int(stop_frame):
        break
    print(frame)

    (success, img) = vid.read()

    if not success:
        break

    # create image for single dot per gaze
    overlay_single_img = np.zeros((img.shape[0], img.shape[1], 3), 'uint8')

    # create image for heatmap that represents the disribution of gazes
    if show_heatmap:
        heatmap = np.zeros((int(img.shape[0] / heatmap_scale), int(img.shape[1] / heatmap_scale)), 'uint8')
        overlay_heatmap_img = np.zeros((img.shape[0], img.shape[1], 3), 'uint8')

    for df_idx, df in enumerate(data_list):
        try:
            slice_ = get_slice(df['frame'], start_idx[df_idx], int(frame))
        except IndexError:
            # end of array
            break
        start_idx[df_idx] = slice_.stop
        x = np.median(df['x'][slice_]) + x_offset
        y = np.median(df['y'][slice_]) + y_offset

        if not np.isnan(x):
            # draw single dots
            cur_color = convert2cvcolor(colors_gaze_dots(df_idx)).astype(np.uint8)
            cur_color = tuple([int(x) for x in cur_color])
            cv2.circle(overlay_single_img, (int(x), int(y)), 15, cur_color) # 15
            pts = cv2.ellipse2Poly((int(x), int(y)), (2, 2), 0, 0, 360, int(360 / 6))
            cv2.fillConvexPoly(overlay_single_img, pts, cur_color)

            # mark position for heatmap
            if show_heatmap:
                cv2.circle(heatmap, (int(x / heatmap_scale), int(y / heatmap_scale)), 1, 1)  # 输入的圆心坐标需要是int类型

    if show_heatmap:
        # process heatmap
        heatmap = np.array(heatmap, 'float32')
        heatmap = ndi.gaussian_filter(heatmap, sigma=heatmap_gauss_sigma / heatmap_scale)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        cv2.normalize(heatmap, heatmap, 0, n_contourlines, cv2.NORM_MINMAX)
        heatmap = np.array(heatmap, 'uint8') % 2

        # find contours
        contours, hierarchy = cv2.findContours(heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # ref: https://www.cnblogs.com/guobin-/p/10842486.html
        hierarchy_levels = get_hierarchy_levels(hierarchy)

        # draw contourlines
        for i in range(len(contours)):
            c2 = cv2.approxPolyDP(contours[i], 2, 0)

            cur_color = convert2cvcolor(colors_contourlines(hierarchy_levels[i])).astype(np.uint8)
            cur_color = tuple([int(x) for x in cur_color])

            cv2.polylines(overlay_heatmap_img, [c2], 3, cur_color, 1)

    # combine images
    final_img = cv2.addWeighted(img, alpha_video, img, 0, 0)

    # combine images: gaze dots
    mask_gaze_dots = np.array(ndi.maximum_filter(overlay_single_img, [1, 1, 5]) != 0, 'uint8')
    mask_gaze_dots = cv2.multiply(final_img, mask_gaze_dots)
    final_img = cv2.addWeighted(final_img, 1, mask_gaze_dots, -alpha_gazes_single, 0)
    final_img = cv2.addWeighted(final_img, 1, overlay_single_img, alpha_gazes_single, 0)

    if show_heatmap:
        # combine images: gaze contour lines
        mask_gaze_contour = np.array(ndi.maximum_filter(overlay_heatmap_img, [1, 1, 5]) != 0, 'uint8')
        mask_gaze_contour = cv2.multiply(final_img, mask_gaze_contour)
        final_img = cv2.addWeighted(final_img, 1, mask_gaze_contour, -alpha_gazes_contourlines, 0)
        final_img = cv2.addWeighted(final_img, 1, overlay_heatmap_img, alpha_gazes_contourlines, 0)

    if is_debug:
        cv2.imshow('gaze on video', final_img)
        cv2.waitKey(1)
    # write image to video writer
    vid_writer.write(final_img)

cv2.destroyAllWindows()