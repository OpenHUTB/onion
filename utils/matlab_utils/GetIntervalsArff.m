% function GetIntervalsArff:
%
% Uses the data loaded from an ARFF file along with an attribute name and type
% of eye movement and returns the intervals for the specific eye movement.
% 使用从带有属性名和眼动类型的ARFF文件中加载数据，并返回特定眼动的间隔。
%
% input:
%   data            - 使用LoadArff脚本从ARFF文件中加载的数据。data loaded from ARFF file with LoadArff
%   arffAttributes  - 从LoadArff脚本返回的属性。 attributes returned from LoadArff
%   attribute       - 在间隔计算中所考虑的属性。 attribute to consider for interval counting
%   moveId          - 所考虑眼动的id。 id for eye movement to consider
%
% output:
%   intervals       - n*6数据（开始时间、开始点x,y；结束时间、结束点x,y）。 nx6 array (start time, start x, start y, end time, end x, end y)

function [intervals] = GetIntervalsArff(data, arffAttributes, attribute, moveId)
    
    intervalIndices = GetIntervalsIndexArff(data, arffAttributes, attribute, moveId);

    % 初始化数据。 initialize data
    intervals = zeros(size(intervalIndices,1),6);

    % 找出数据中属性的位置。 find position of attribute in data
    timeIndex = GetAttPositionArff(arffAttributes, 'time');
    xIndex = GetAttPositionArff(arffAttributes, 'x');
    yIndex = GetAttPositionArff(arffAttributes, 'y');

    for i=1:size(intervals,1)
        startIndex = intervalIndices(i,1);
        endIndex = intervalIndices(i,2);
        intervals(i,:) = [data(startIndex,timeIndex) data(startIndex,xIndex) data(startIndex,yIndex) data(endIndex,timeIndex) data(endIndex,xIndex) data(endIndex,yIndex)];
    end
end    
