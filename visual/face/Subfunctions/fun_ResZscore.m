%  z-score 是对某一原始分值进行转换，变成的一个标准分值，该标准分值可使得原来无法比较的数值变得可比。
% 变成一个标准分数，使随机网络生成的激活可以比较
% (一个特定的观察结果 - 平均数) / 标准差　＝　该观察结果的z分数
% 负 z 分数表示低于平均数的值；正 z 分数表示高于平均数的值。
function [resp_obj_z, resp_obj_z_3D] = fun_ResZscore(resp_rand, num_cell, face_idx, numCLS, numIMG)

resp_reshape = reshape(resp_rand, num_cell, numCLS*numIMG);

resp_obj = resp_reshape(face_idx, 1:numCLS*numIMG);
resp_obj_3D = zeros(length(face_idx), numCLS, numIMG);
for ii = 1:numCLS
    resp_obj_ii = resp_obj(:,(ii-1)*numIMG+1:numIMG*ii);
    resp_obj_3D(:,ii,:) = resp_obj_ii;
end

resp_obj_z = zeros(length(face_idx), numCLS*numIMG);
resp_obj_z_3D = zeros(length(face_idx), numCLS,numIMG);
resp_obj_max_z = zeros(length(face_idx), numIMG);
for ii = 1:length(face_idx)
    resp_obj_ii = resp_obj(ii,:);
    resp_obj_3D_ii = squeeze(resp_obj_3D(ii,:,:));
    [~,max_obj_idx] = max(mean(resp_obj_3D_ii(2:end,:),2));
    norm_mean = mean(resp_obj_3D_ii(max_obj_idx+1,:));  % 求平均值
    norm_std = std(resp_obj_3D_ii(max_obj_idx+1,:));
    resp_obj_z(ii,:) = (resp_obj_ii-norm_mean)/norm_std;  % (x-平均值)/标准差
    resp_obj_z_3D(ii,:,:) = (resp_obj_3D_ii-norm_mean)/norm_std;
    resp_obj_max_z(ii,:) = (resp_obj_3D_ii(max_obj_idx+1,:)-norm_mean)/norm_std;
end

end