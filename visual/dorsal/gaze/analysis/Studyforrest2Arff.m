% Studyforrest2Arff.m
%
% This function converts gaze data from studyforrest file format to ARFF. The
% input comprises from a gaze file and a frames timing file
%
% input:
%   eventsFile  - file containing events (onset, duration, frameidx, videotime, audiotime, lasttrigger)
%   gazeFile    - file containing gaze (x, y, pupil dilation(units?), frameId reference)
%   metadata    - metadata in the form accepted from SaveArff.m
%   outputFile  - (optional) name of ARFF. If it is not used the data is stored in the same directory as events with .arff extension

function Studyforrest2Arff(eventsFile, gazeFile, metadata, outputFile)
    if (nargin < 4)
        [dir, file, ext] = fileparts(eventsFile);
        if (length(dir) > 0)
            dir = [dir '/'];
        end

        ind = findstr(file, '_events');

        outputFile = [dir file(1:ind(end)-1) '.arff'];
    end

    frames = importdata(eventsFile, '\t');
    frames = frames.data;
    % check for missing frames
    %{
    frame_step = frames(2:end,3) - frames(1:end-1,3);
    if (size(find(frame_step>1),1) > 0)
        warning(['Missing frames in ' eventsFile '. No ARFF file is written']);
        return;
    end
    %}

    gaze = importdata(gazeFile, '\t');
    if (IsOctave())
        % at points where confidence is 0 we have 2 entries in the .tsv file.
        % This translates to 2 columns from importdata with data. The last 2
        % columns are zeros. Below we add the frame numbering to the last
        % column. Also when the 3rd column is 0 the confidence is 0 too
        gaze(gaze(:,3)==0,4) = gaze(gaze(:,3)==0,2);
        gaze(gaze(:,3)==0,2) = 0; % remove frameId values
    else
        % in matlab the first two columns are NaN
        gaze(isnan(gaze)) = 0;
    end

    % get gaze points only until the last frame
    if (gaze(end,4) > frames(end,3))
       warning(['Gaze entries point to more frames than they actually exist in ' eventsFile]);
       gaze = gaze(gaze(:,4)<=frames(end,3),:);
    end
    

    % allocate and assign arff values
    arffData = zeros(size(gaze,1),5); % (time, x, y, confidence, frame_id)
    arffData(:,2) = gaze(:,1); % x
    arffData(:,3) = gaze(:,2); % y
    arffData(:,4) = 1.0; % confidence
    arffData(gaze(:,3)==0,4) = 0.0;
    arffData(:,5) = gaze(:,4); % frame id

    % calculate timestamps
    % get gaze count for each frame
    gazeCount = histc(gaze(:,4), frames(:,3));

    prevFrameId = -1;
    gazeCounter = 0;
    gazeStep = 0;
    for i=1:size(arffData,1)
       frameId = gaze(i,4);
       if (frameId ~= prevFrameId)
           gazeCounter = 0;
           gazeStep = frames(frameId,2)/gazeCount(frameId);
       end
       arffData(i,1) = frames(frameId,1) + gazeCounter*gazeStep;
       % make sure timestamps are monotonous. i.e. we don't move past next frame's time
       if (frameId<size(frames,1) && arffData(i,1) > frames(frameId+1,1))
            arffData(i,1) = frames(frameId+1,1);
       end

       % convert to microseconds
       arffData(i,1) = arffData(i,1)*1000000;

       gazeCounter = gazeCounter + 1;
       prevFrameId = frameId;
    end
    arffData(:,1) = floor(arffData(:,1)); % round to int

    % if there is a difference in the amount of frames referenced in gazeFile
    % and those in eventsFile then we get inf and nan values. The solution is to
    % keep entries just before those observations appear. (The reason for the frame
    % difference is not clear)
    indInf = find(isinf(arffData(:,1)));
    indNan = find(isnan(arffData(:,1)));
    indTot = min([indInf; indNan]);
    
    if (size(indTot,1)>0)
        arffData = arffData(1:indTot-1,:);
        warning(['Missing frames in ' eventsFile '. Written timestamps might be erroneous']);
    end

    % save data
    relation = 'studyforrest_gaze';

    attributes = {'time', 'INTEGER';
                  'x', 'NUMERIC';
                  'y', 'NUMERIC';
                  'confidence', 'NUMERIC';
                  'frame_id', 'INTEGER'};

    %SaveArff(outputFile, arffData, metadata, attributes, relation);

	% write to file directly to speed up the process. 7 times faster based on measurements
	% start writing
    fid = fopen(outputFile, 'w+');

    % write relation
    fprintf(fid, '@RELATION %s\n\n', relation);

    % write metadata
    fprintf(fid, '%%@METADATA width_px %d\n', metadata.width_px);
    fprintf(fid, '%%@METADATA height_px %d\n', metadata.height_px);
    fprintf(fid, '%%@METADATA width_mm %.2f\n', metadata.width_mm);
    fprintf(fid, '%%@METADATA height_mm %.2f\n', metadata.height_mm);
    fprintf(fid, '%%@METADATA distance_mm %.2f\n\n', metadata.distance_mm);

    % write metadata extras. Those are data that vary between experiments
    for i=1:size(metadata.extra,1)
        fprintf(fid, '%%@METADATA %s %s\n', metadata.extra{i,1}, metadata.extra{i,2});
    end
    % print an empty line
    fprintf(fid, '\n');
	for i=1:size(attributes,1)
        fprintf(fid, '@ATTRIBUTE %s %s\n', attributes{i,1}, upper(attributes{i,2}));
    end

    % write data keyword
    fprintf(fid,'\n@DATA\n');
    % write actual data
    %for i=1:size(arffData,1)
    %    fprintf(fid, '%d,%.2f,%.2f,%.2f,%d\n', arffData(i,:));
    %end
    fprintf(fid, '%d,%.2f,%.2f,%.2f,%d\n', arffData'); % simpler and faster

    % close file
    fclose(fid);
end
