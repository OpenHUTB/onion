% Arff2Coord.m
%
% This function converts an ARFF file to its .coord representation.
%
% input:
%   arrfFile    - input ARFF file
%   coordFile   - coord file to sore results

function Arff2Coord(arffFile, coordFile)
    [data, metadata, attributes, relation] = LoadArff(arffFile);

    timeInd = GetAttPositionArff(attributes, 'time');
    xInd = GetAttPositionArff(attributes, 'x');
    yInd = GetAttPositionArff(attributes, 'y');
    confInd = GetAttPositionArff(attributes, 'confidence');

    fid = fopen(coordFile, 'w');
    fprintf(fid, 'gaze %d %d\n', metadata.width_px, metadata.height_px);
    fprintf(fid, 'geometry distance %.2f width %.2f height %.2f\n', metadata.distance_mm/1000, metadata.width_mm/1000, metadata.height_mm/1000);
    for i=1:size(data,1)
        fprintf(fid, '%d %.2f %.2f %.2f\n', data(i,timeInd), data(i,xInd), data(i,yInd), data(i, confInd));
    end
    fclose(fid);
end
