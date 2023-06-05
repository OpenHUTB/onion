% 在A中查找item
% 如果不存在，idx=0；
% 如果存在曾返回对应A中的索引idx
function idx = row_index(A, item)
idx = 0;
for i = 1 : size(A, 1)
    if sum(A(i, :) == item) == size(A, 2)
        idx = i;
        return;
    end
end

end