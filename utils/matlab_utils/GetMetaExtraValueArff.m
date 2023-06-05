% GetMetaExtraValueArff.m
%
% This function returns the value of a metadata stored in extra metadta as
% string.

function value = GetMetaExtraValueArff(arffMeta, metaName)
    metaInd = GetMetaExtraPosArff(arffMeta, metaName);

    value = arffMeta.extra{metaInd,2};
end
