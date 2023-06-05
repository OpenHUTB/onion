function [cell_idx] = fun_FindNeuron(act_rand,num_cell,numCLS,numIMG,pThr,indClass)

act_reshape = reshape(act_rand, num_cell, numCLS*numIMG);  % 每张图对应一个43264向量; 细胞数43264 * （类别数6 * 每一类200张图像）
act_3D = zeros(num_cell, numCLS, numIMG);  % 43264 * 6 * 200

for cc = 1:numCLS
    act_3D(:, cc, :) = act_reshape(:, (cc-1)*numIMG+1 : cc*numIMG);  % 每次 43264*200
end

pref_class = zeros(num_cell, 1);  % 每一个细胞偏好的类别: 43264*1
sel_p_val = zeros(num_cell, 1);   % 每一个细胞所对应的 p 值: 43264*1

% parfor
for cc = 1 : num_cell  % parfor
    mean_FR = [];  % 对特定位置200张图片激活的平均（Face Response）
    for cls = 1 : numCLS
        mean_FR = [mean_FR, mean(act_reshape(cc, (cls-1)*numIMG+1 : cls*numIMG))];  % 每个类别所有200张图片对应一个mean_FR（对200张图像该位置的激活求平均）
    end
    
    [~,sort_ind] = sort(mean_FR, 'descend');  % 当前细胞位置，对6个类别的平均激活进行降序排列(当然位置激活最大的类别id为偏好类别id)
    pref_class(cc) = sort_ind(1);             % 选取6个类别中激活最大的类别作为当前细胞位置偏好的类别
    
    resp1 = act_reshape(cc, (sort_ind(1)-1)*numIMG+1:numIMG*sort_ind(1) );  % 偏好类别图像(200 张)在当前位置cc的激活resp1 (1*200)
    pval_temp = [];
    for ee = 2:numCLS
        resp2 = act_reshape(cc, (sort_ind(ee)-1)*numIMG+1:numIMG*sort_ind(ee));  % 除了偏好类别的其他类别图像(每个200张) 激活(resp2) 1*200
        pval_temp(ee-1) = ranksum(resp1, resp2);  % 相对于偏好类别 的p值
    end
    sel_p_val(cc) = max(pval_temp);  % 当前位置选是该类别选择单元的p值 <- 1*5
end

for ii = indClass  % 类别偏好为脸(idxClass: 1) 且 对应p小于0.001 的细胞有468个
    cell_idx = find((pref_class==ii) & (sel_p_val<pThr));  % 除了偏好类别的其他类别 p 值的最大值 必须小于 pThr (p 值必须足够小, 可信度高)
end

end