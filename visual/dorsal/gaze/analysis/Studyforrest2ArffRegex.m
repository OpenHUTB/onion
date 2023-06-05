% function Studyforrest2ArffRegex:
% gets as inputs regular expressions to gaze and events files. 
% The execution of this function is around 3 times faster in matlab
%
% input:
%   eventsFiles - regex to events files
%   gazeFiles   - regex to gaze files
%   outputDir   - directory to save results (.arff)
%
% ex. fMRI results
% Studyforrest2ArffRegex('/path/to/studyforrest/*/*/*/*movie*events.tsv', '/path/to/studyforrest/*/*/*/*eyegaze_physio.tsv', '/path/to/results');
%
% Studyforrest2ArffRegex('/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-1_events.tsv', '/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv', '/data2/whd/workspace/sot/hart/result/gaze_data');
% Studyforrest2ArffRegex('/data2/whd/workspace/data/neuro/software/ds000113/*/*/*/*movie*events.tsv', '/data2/whd/workspace/data/neuro/software/ds000113/*/*/*/*eyegaze_physio.tsv', '/data2/whd/workspace/sot/hart/result/gaze_data');

function Studyforrest2ArffRegex(eventFiles, gazeFiles, outputDir)
    % set metadata for fMRI and lab experiments
    metadataFmri.width_px = 1280;
    metadataFmri.height_px = 546;
    metadataFmri.width_mm = 265;
    metadataFmri.height_mm = 113;
    metadataFmri.distance_mm = 630;
    metadataFmri.extra = {};

    metadataLab.width_px = 1280;
    metadataLab.height_px = 546;
    metadataLab.width_mm = 522;
    metadataLab.height_mm = 223;
    metadataLab.distance_mm = 850;
    metadataLab.extra = {};

    eventFilelist = glob(eventFiles);
    gazeFilelist = glob(gazeFiles);

    assert(size(eventFilelist,1)==size(gazeFilelist,1), 'Provided regular expressions returned different number of files');

    % Subjects 1 to 20 were recorded in the scanner. Subjects 21 to 36 were recorded in the lab
    for fileId=1:size(eventFilelist,1)
        disp(sprintf('%s\n%s\n\n', eventFilelist{fileId,1}, gazeFilelist{fileId,1}));
        % get subject id
        ind = findstr(eventFilelist{fileId,1}, 'sub-');
        subId = str2num(eventFilelist{fileId,1}(ind+4:ind+5));

        [dir, name, ext] = fileparts(eventFilelist{fileId,1});
        outputFile = fullfile (outputDir, [name '.arff']);

        if (subId <= 20)
            disp('fmri')
            Studyforrest2Arff(eventFilelist{fileId,1}, gazeFilelist{fileId,1}, metadataFmri, outputFile);
        else
            disp('lab')
            Studyforrest2Arff(eventFilelist{fileId,1}, gazeFilelist{fileId,1}, metadataLab, outputFile);

            % put in-scanner metadata in order to be able to cluster everything together
            %Studyforrest2Arff(eventFilelist{fileId,1}, gazeFilelist{fileId,1}, metadataFmri);
        end
    end
end
