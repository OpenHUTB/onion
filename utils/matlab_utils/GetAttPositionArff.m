% function GetAttPositionArff:
%
% Gets a list of attributes returned from LoadArff and an attribute name to
% search.  If it finds the attribute returns its index otherwise it can raise
% an error.
% 从LoadArff返回的属性列表中搜索属性的索引，找到则返回它的的索引，否则返回错误。
%
% input:
%   arffAttributes  - 从LoadArff函数中返回的属性列表。attribute list returned from LoadArff
%   attribute       - 需要搜索的属性。attribute to search
%   check           - （可选）是否检查搜索的属性是否存在。(optional) boolean to check if attribute exists. Default is true
%
% output:
%   attIndex        - 如果找到则返回属性的索引属性。index attribute of the attribute in the list if it was found. 
%                     如果没找到返回0。 Returns 0 if it wasn't found

function [attIndex] = GetAttPositionArff(arffAttributes, attribute, check)
    if (nargin < 3)
        check = true;
    end
    attIndex = 0;

    for i=1:size(arffAttributes,1)
        if (strcmpi(arffAttributes{i,1}, attribute) == 1)
            attIndex = i;
        end
    end

    % check index
    if (check)
        assert(attIndex>0, ['Attribute "' attribute '" not found']);
    end
end
