%%% mask2coord
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cor, intensity, cor_singlecolumn,M,DIM,TF,df] = mask2coord(mask, checkdimension)
% [cor, intensity, cor_singlecolumn] = mask2coord(mask, checkdimension)
%
% This is to retrieve the coordinate of a mask file, or a matrix of 3-D
%
% mask: an image file or a matrix (3-D), with some of the elements are
% non-zeros
% checkdimension: check if the dimension is checkdimension, if not, return empty
% matrix
% cor: a N*3 matrix, which each row a coordinate in matrix
% intensity: a N*1 matrix, which encodes the intensity of each voxel.
% cor_singlecolumn: a N*1 matrix, each row is the index in the matrix
% M: rotation matrix
% DIM: dimension
% TF: t test or f test? 'T','F' or []
% df: degree of freedome for T/F test
%
% Example:
%   mask = zeros(4,3,2);
%   mask(1,2,1) = 1;
%   mask(3,2,2) = 1;
%   mask2coord(mask)
%
%   mask2coord('spmT_0002.img')
%   mask2coord('spmT_0002.img',[41 48 35])
%
% Xu Cui
% 2004-9-20
% last revised: 2005-04-30

if nargin < 2
    checkdimension = 0;
end

if ischar(mask)
    V = spm_vol(mask);
    mask = spm_read_vols(V);
    M = V.mat;
    DIM = V.dim;
    TF = 'T';
	T_start = strfind(V.descrip,'SPM{T_[')+length('SPM{T_[');
    if isempty(T_start); T_start = strfind(V.descrip,'SPM{F_[')+length('SPM{F_['); TF='F'; end
    if isempty(T_start)
        TF=[]; df=[];
    else
    	T_end = strfind(V.descrip,']}')-1;
    	df = str2num(V.descrip(T_start:T_end));    
    end
else
    M = [];
    TF = [];
    df = [];
end

dim = size(mask);
if length(checkdimension)==3
    if dim(1)~= checkdimension(1) | dim(2)~= checkdimension(2) | dim(3)~= checkdimension(3)
        y = [];
        disp('dimension doesn''t match')
        return
    end
end

pos = find(mask~=0);    % mask中不为0的位置
intensity = mask(pos);  % mask中不为0的所有强度的集合

y = zeros(length(pos),3);

y(:,3) = ceil(pos/(dim(1)*dim(2)));
pos = pos - (y(:,3)-1)*(dim(1)*dim(2));
y(:,2) = ceil(pos/dim(1));
pos = pos - (y(:,2)-1)*(dim(1));
y(:,1) = pos;

cor = y;
cor_singlecolumn = pos;
DIM = dim;
return
