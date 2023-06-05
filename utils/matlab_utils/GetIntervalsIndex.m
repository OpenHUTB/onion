% function GetIntervalsIndex:
%
% Uses the data along with a data value and returns the intervals (as indices) 
% for the provided value
%
% input:
%   data       - is a data vector
%   value      - value to use for intervals calculation
%
% output:
%   intervals  - nx2 array (start index, end index)

function [intervals] = GetIntervalsIndex(data, value)
    assert(size(data,1) == 1 | size(data,2) == 1, 'Data should be a vector');
    % initialize data
    intervals = zeros(0,2);

    % find position of attribute in data

    startIndex = -1;
    for i=1:size(data,1)
        if (data(i)==value)
            % first element of interval
            if (startIndex==-1)
                startIndex=i;
            end
        else
            % interval finished on previous iteration
            if (startIndex~=-1)
                intervals = [intervals; startIndex i-1];
            end
            startIndex = -1;
        end
    end
    % add last interval
    if (startIndex~=-1)
        intervals = [intervals; startIndex length(data)];
    end
end    
