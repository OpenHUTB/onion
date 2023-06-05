% ComputeRegressors.m
%
% This function gets an ARFF as input and segments it in intervals. For each
% interval we assign a SP value and a saccade value.
% 
% input:
%   arffFile    - arffFile to load
%   attName     - attribute name to use for eye movement intervals
%   saccShare   - saccade percentage for given subject
%   spShare     - Sp percentage for given subject
%
% output:
%   intStart    - start time of intervals
%   intDur      - duration of intervals
%   spValues    - parameter values for SP
%   saccValues  - parameter values for sacade

function [intStart, intDur, spValues, saccValues] = ComputeRegressors(arffFile, attName, saccShare, spShare)
    c_blockDur = 2000000; % 2 seconds
    c_spShare = 5*spShare; 
    c_saccShare = 1.5*saccShare; 
	
    [data, metadata, attributes] = LoadArff(arffFile);
    timeInd = GetAttPositionArff(attributes, 'time');

    spIntervals = GetIntervalsArff(data, attributes, attName, 3);
    saccIntervals = GetIntervalsArff(data, attributes, attName, 2);

    spIntervals = ConcatenateIntervals(spIntervals);
    spIntervals = ConcatenateIntervals(spIntervals);

    saccIntervals = ClearInts(saccIntervals, spIntervals);

	spDur = spIntervals(:,4) - spIntervals(:,1);
    saccDur = saccIntervals(:,4) - saccIntervals(:,1);

	curTime = 0;
    recTime = data(end,timeInd);
    intStart = [];
    intDur = [];
    spValues = [];
    saccValues = [];
    while (curTime < recTime)
        spValidInts = (spIntervals(:,1) > curTime & spIntervals(:,1) < curTime+c_blockDur) | ...
                      (spIntervals(:,4) > curTime & spIntervals(:,4) < curTime+c_blockDur);
        spTmpInts = spIntervals(spValidInts,:);
        spTmpInts(spTmpInts(:,4) > curTime+c_blockDur,4) = curTime+c_blockDur;
        spTmpInts(spTmpInts(:,1) < curTime,1) = curTime;
        spTmpDur = spTmpInts(:,4) - spTmpInts(:,1);

        saccValidInts = saccIntervals(:,1) > curTime & saccIntervals(:,1) < curTime+c_blockDur;

        spShare = sum(spTmpDur) / c_blockDur;
        spIntNum = sum(spValidInts);

        saccShare = sum(saccDur(saccValidInts)) / c_blockDur;
        saccIntNum = sum(saccValidInts);
        intStart = [intStart; curTime];
        intDur = [intDur; c_blockDur];
        spValues = [spValues; spShare/c_spShare];
        saccValues = [saccValues; saccShare/c_saccShare];

        curTime = curTime + c_blockDur;
    end

    spValues = spValues - mean(spValues);
    spValues(spValues > 1) = 1;
    spValues(spValues < -1) = -1;

    saccValues = saccValues - mean(saccValues);
    saccValues(saccValues > 1) = 1;
    saccValues(saccValues < -1) = -1;

    function l_ints = ConcatenateIntervals(intervals)
        c_spGap = 100000; % 100ms
    	l_ints = [];
        duration = intervals(:,4) - intervals(:,1);
        timeGap = intervals(2:end,1) - intervals(1:end-1,4);
        first = 1;
        second = 1;
        for i=1:size(intervals,1)-1
            if (timeGap(i) < c_spGap || duration(i+1) > timeGap(i) || duration(i) > timeGap(i))
                second = i+1;
            else
                l_ints = [l_ints; intervals(first,1:3) intervals(second, 4:6)];
                first = i+1;
                second = i+1;
            end
        end
    end

    function saccInts = ClearInts(saccInts, spInts)
        c_displace = 100000;
        for i = 1:size(spInts,1)
            saccInts(saccInts(:,1) > spInts(i,1) - c_displace & saccInts(:,1) < spInts(i,4) + c_displace,    :) = [];
            saccInts(saccInts(:,4) > spInts(i,1) - c_displace & saccInts(:,4) < spInts(i,4) + c_displace,    :) = [];
        end
    end

end
