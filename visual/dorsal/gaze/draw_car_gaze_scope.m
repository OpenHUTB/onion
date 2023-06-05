% 在图片上绘制凝视区域
img_dir = 'C:\BaiduSyncdisk\project\青基\figs\raw';
I = imread(fullfile(img_dir, '2023-03-04 11-18-16.jpg'));
% imshow(I);

%%
load('PedNet.mat');
PedNet.Layers

% Load an input image.
im = imresize(I,[480,640]);
% imshow(im);

addpath(fullfile(matlabroot, 'examples', 'deeplearning_shared', 'main'));
% ped_bboxes = pedDetect_predict(im);

ped_bboxes = [183, 160, 260, 230;
    433, 195, 90, 90;
    535, 215, 40, 35];

% 在未缩放得图上进行显示
height_scale = size(I, 1) / size(im, 1);
width_scale = size(I, 2) / size(im, 2);
ped_bboxes_pre = [ped_bboxes(:, 1) * width_scale, ped_bboxes(:, 2) * height_scale, ...
                  ped_bboxes(:, 3) * width_scale, ped_bboxes(:, 4) * height_scale];
pos = [ped_bboxes_pre(:,1), ped_bboxes_pre(:, 2)-80, ped_bboxes_pre(:,3) ped_bboxes_pre(:,4)];
cur_color = [0.4660 0.6740 0.1880]*256;
outputImage = insertShape(I, ...
    'Rectangle', pos, ...  % 框向上调了80个像素
    'Color', cur_color, ...
    'LineWidth',3);
label_str = 'Car';
outputImage = insertObjectAnnotation(outputImage, "rectangle", pos, label_str, ...
    Color=cur_color, ...
    TextBoxOpacity=0.9,FontSize=18);
imshow(outputImage);

%%
% 'InteractionsAllowed', 'none': 圆圈上没有点
center_point = [(ped_bboxes(:, 1) + ped_bboxes(:, 3)/2) * width_scale + 50, ...
                ped_bboxes_pre(:, 2) + 100
               ];
all_radius = [1, 40:50:40+50*3];
all_colors = [
    [0.9 0.9 0.9]; ...  % 中间为亮灰色（RGB相同为灰色），越往外越深
    [0.8 0.8 0.8]; ...
    [0.6 0.6 0.6]; ...
    [0.4 0.4 0.4]; ...
    [0.3 0.3 0.3]
];
all_width = [3.5, 3, 2.5, 2, 1];

for i = 1 : numel(all_radius)
    roi = images.roi.Circle(gca, ...
        'Center', center_point(1, :), ...
        'Radius', all_radius(i), ...
        'Color', all_colors(i, :), ...
        'LineWidth', all_width(i), ...
        'InteractionsAllowed', 'none' ...  % 圆圈上没有点
    );
end



