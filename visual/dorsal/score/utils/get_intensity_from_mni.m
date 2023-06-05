% 从fMRI强度集合中(x,y,z,intensity)中查找特定坐标下(xi,yi,zi)的强度值 （未用）
% 使用HashMap加快检索速度
% 使用一个long类型的整数，前32位存储x坐标，后32位存储y坐标
% 先将int型的x和y坐标都转换成long，然后x左移32位；
% long temp=(long)x<<32+(long)y;
% 参考: https://blog.csdn.net/kfz2394552181/article/details/122824170

% 如果不存在，intensity=0；
% 如果存在曾返回对应mni_intensity中的强度值
function intensity = get_intensity_from_mni(mni_intensity, pos)
% intensity = 0;
% for i = 1 : size(mni_intensity, 1)
%     if sum(mni_intensity(i, :) == pos) == size(mni_intensity, 2)
%         intensity = i;
%         return;
%     end
% end

end