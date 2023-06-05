% ppd2distance.m
%
% This function convert pixels per degree to a pair of two distances for a
% monitor based experiment. It takes as input the distance from the monitor and
% returns the second dimension
%
% input:
%   ppd         - pixels per degree
%   pixNum      - number of pixels 
%   distance    - distance from monitor. This can be any number since the system is
%                 underdetermined
%
% output:
%   monitorDim  - monitor size for the provided data

function monitorDim = ppd2distance(ppd, pixNum, distance)
    thetaDeg = pixNum / ppd;
    thetaRad = thetaDeg * pi / 180;

    monitorDim = 2 * distance * tan(thetaRad / 2);
end
