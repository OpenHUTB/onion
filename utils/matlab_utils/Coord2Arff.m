% function Coord2Arff:
i%
% Converts coord files to ARFF files. The created file has the same name
% with .arff extension.
%
% input:
%   coordFile       - .coord file
%   storeAtSource   - 1 to store .arff at the same position as .coord. 0 to store at current directory

function Coord2Arff(coordFile, storeAtSource)
    % check input arguments
    if (nargin <2)
        storeAtSource = 0;
    end
    % get part of input file
    [dir, basename, ext] = fileparts(coordFile);

    % load data from coord file
    [data, pixelX, pixelY, width, height, distance, conf] = LoadCoordAndConf(coordFile);

    arffFile = [basename '.arff'];
    if (length(dir) > 0)
        dir = [dir '/'];
    end
    if (storeAtSource == 1)
        arffFile = [dir arffFile];
    end

    fid = fopen(arffFile, 'w+');
    % print experiment parameters
    fprintf(fid, '@RELATION gaze_labels\n\n');
    fprintf(fid, '%%@METADATA width_px %d\n', pixelX);
    fprintf(fid, '%%@METADATA height_px %d\n', pixelY);
    fprintf(fid, '%%@METADATA distance_mm %.2f\n', distance*1000);
    fprintf(fid, '%%@METADATA width_mm %.2f\n', width*1000);
    fprintf(fid, '%%@METADATA height_mm %.2f\n\n', height*1000);

    fprintf(fid, '@ATTRIBUTE time INTEGER\n');
    fprintf(fid, '@ATTRIBUTE x NUMERIC\n');
    fprintf(fid, '@ATTRIBUTE y NUMERIC\n');
    fprintf(fid, '@ATTRIBUTE confidence NUMERIC\n\n');
    fprintf(fid, '@DATA\n');

    for i=1:size(data,1)
        fprintf(fid, '%d,', data(i,1));
        fprintf(fid, '%.2f,', data(i,2:3));
        fprintf(fid, '%.2f\n', conf(i));
    end

    fclose(fid);
end
