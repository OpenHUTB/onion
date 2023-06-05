% 支持函数 helperPlotMap 生成输入图像和指定的可解释性图的图。
function helperPlotMaps(img,YPred,YTrue,maps,mapNames,viewingAngle,mapClass)
nexttile
surf(img,EdgeColor="none")
view(viewingAngle)
title({"True: "+ string(YTrue), "Predicted: " + string(YPred)}, ...
    interpreter="none")
colormap parula

numMaps = length(maps);
for i = 1:numMaps
    map = maps{i};
    mapName = mapNames(i);

    nexttile
    surf(map,EdgeColor="none")
    view(viewingAngle)
    title(mapName,mapClass,interpreter="none")
end
end