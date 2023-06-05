% GetMetaExtraPosArff.m
%
% This function finds the position of a metadata stored in the extra metadata.
% The default metadata are width_px, height_px, height_mm, width_mm, distance_mm
% are accessed as fields of the structure.
%
% input:
%   arffMeta    - structure holding the metadata
%   metaName    - name of the metadata to find
%
% output:
%   metaIndex   - index in the metadata field extra which is an nx2 cell array

function metaIndex = GetMetaExtraPosArff(arffMeta, metaName)
    metaIndex = 0;
    for i=1:size(arffMeta.extra,1)
        if(strcmpi(arffMeta.extra{i,1}, metaName) == 1)
            metaIndex = i;
        end
    end

    assert(metaIndex > 0, ['Extra metadata ' metaName ' not found']);
end
