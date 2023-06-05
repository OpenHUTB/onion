% ComputeRegressors.m.m
%
% This function gets an ARFF file as input and segments it in intervals. For each
% interval we assign a SP value and a saccade value and motion value from the video
% 该函数使用run_detection.py的输出ARFF文件作为输入，并按间隔进行分割。
% 对于每个间隔我们从视频中分配一个平稳跟踪的值、一个眼跳值和一个运动值。
% 
% input:
%   arffFile    - arffFile to load
%   attName     - attribute name to use for eye movement intervals
%   saccShare   - 对应受试者的眼跳百分比。saccade percentage for given subject
%   spShare     - 对应受试者平稳跟踪的百分比。Sp percentage for given subject
%   motionFile  - 文件包含的运动信息。file containing motion information
%
% output:
%   intStart    - start time of intervals
%   intDur      - duration of intervals
%   spValues    - 平稳跟踪的参数值。parameter values for SP
%   saccValues  - parameter values for sacade
%   motionValues- parameter values for motion

function [intStart, intDur, spValues, saccValues, motionValues] = ComputeRegressors(arffFile, attName, saccShare, spShare, motionFile)
    c_blockDur = 2000000; % 每两个之间的间隔时间为2秒。  2 seconds
    % 受试者影评问跟踪的百分比 乘以5（为了反映平稳跟踪运动中较大的方差，将平稳跟踪的调制值设置为5），
    % 为了保证带来调制值，2秒时间窗口内平稳跟踪正则化的数据大约有95%在1以下。
    c_spShare = 5*spShare;  %  图像比较尖，乘以一个较大值，以拉近和眼跳之间的分布。
    c_saccShare = 1.5*saccShare;
    c_motionHigh = 9.2859; % 帧运动的百分之90。 90th percentile of frame motion

    motionData = importdata(motionFile);
    c_fps = 25;  % 视频的帧频为25

    [data, metadata, attributes] = LoadArff(arffFile);  % 从gaze_data_with_em中加载run_detction.py检测的平滑跟踪结果。
    timeInd = GetAttPositionArff(attributes, 'time');   % 从LoadArff返回的属性列表中搜索属性的索引

    spIntervals = GetIntervalsArff(data, attributes, attName, 3);
    saccIntervals = GetIntervalsArff(data, attributes, attName, 2);

    spIntervals = ConcatenateIntervals(spIntervals);
    spIntervals = ConcatenateIntervals(spIntervals);

    saccIntervals = ClearInts(saccIntervals, spIntervals);

	spDur = spIntervals(:,4) - spIntervals(:,1);  % 平稳跟踪的持续时间 = 结束时间-开始时间
    saccDur = saccIntervals(:,4) - saccIntervals(:,1);

	curTime = 0;
    recTime = data(end,timeInd);  % 最后一行的第一个元素
    intStart = [];
    intDur = [];
    spValues = [];
    saccValues = [];
    motionValues = [];
    while (curTime < recTime && curTime < 1000000 * size(motionData,1) / c_fps)
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

        % get motion values
        startFrame = round(0.5 + c_fps * curTime/1000000);  % 微秒->秒
        endFrame = round(0.5 + c_fps * (curTime+c_blockDur)/1000000);
        if (startFrame > size(motionData,1))
            break;
        end
        if (endFrame > size(motionData,1))
            endFrame = size(motionData,1);
        end
        motionTmp = motionData(startFrame:endFrame, 1);
        motionSpeedMean = mean(motionTmp(:,1)) / c_motionHigh;  % 所有帧的平均速度，除以9.2859的原因？

        motionValues = [motionValues; motionSpeedMean];

        curTime = curTime + c_blockDur;
    end

    spValues = spValues - mean(spValues);  % 每个平稳跟踪的值减去所有平稳跟踪的平均值
    spValues(spValues > 1) = 1;            % 将平稳跟踪的值限制在[-1, 1]内
    spValues(spValues < -1) = -1;

    saccValues = saccValues - mean(saccValues);
    saccValues(saccValues > 1) = 1;
    saccValues(saccValues < -1) = -1;

    % 不考虑运动中的极端值（这在epicFlow算法中比较常见）。 do not take extreme values (which are common with epicFlow) into account
    motionValues = motionValues - median(motionValues);  % 为什么用中位数而不是平均值
    motionValues(motionValues > 1) = 1;
    motionValues(motionValues < -1) = -1;

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
