%% get image file information
% extract from xjview.m
function [imageFile,M,DIM,TF,df,mni,intensity] = get_image_file(thisfilename)
% function imageFile = getImageFile(filename)
% get the information of this/these image file(s)
% 获得镜像文件的MNI坐标mni和对应的强度信息
%
% thisfilename: (optional) if doesn't give file name, then show a open file
% dialog
% imageFile: the full name of the selected file (if user pressed cancel,
% imageFile == 0
% M: 旋转矩阵（用来干嘛）M matrix (rotation matrix)
% DIM: dimension
% TF: t test or f test? 'T' or 'F'
% df: degree of freedome
% mni: mni coord
% intensity: intensity of each mni coord
%
% Note: The returned variables are cellarrays.
%
% Xu Cui
% last revised: 2005-05-03

if nargin < 1 | isempty(thisfilename)
    if findstr('SPM2',spm('ver'))
        P0 = spm_get([0:100],'*IMAGE','Select image files');
    else%if findstr('SPM5',spm('ver'))
        P0 = spm_select(Inf,'image','Select image files');
    end    
    try
        if isempty(P0)
            imageFile = '';M=[];DIM=[];TF=[];df=[];mni=[];intensity=[];
            return
        end
    end
    for ii=1:size(P0,1)
        P{ii} = deblank(P0(ii,:));
    end
else
    if isstr(thisfilename)
        P = {thisfilename};
    elseif iscellstr(thisfilename)
        P = thisfilename;
    else
        disp('Error: In getImageFile: I don''t understand the input.');
        imageFile = '';M=[];DIM=[];TF=[];df=[];mni=[];intensity=[];
        return
    end
end

global LEFTRIGHTFLIP_;

for ii=1:length(P)
    imageFile{ii} = P{ii}; 
	[cor{ii}, intensity{ii}, tmp{ii}, M{ii}, DIM{ii}, TF{ii}, df{ii}] = mask2coord(imageFile{ii}, 0);
    if LEFTRIGHTFLIP_ == 1
        M{ii}(1,:) = -M{ii}(1,:);
    end
	mni{ii} = cor2mni(cor{ii}, M{ii});
end
