% ComputeRegressorsAll.m
%
% This function calls the ComputeRegressors for all the files in the regular
% and saves the regressors in files with _sp.txt and _fix.txt endings
%
% input:
%   regex   - regular expresion to .arff files
%   attName - attribute name to use for intervals extraction
%   outDir  - dir to save blocks
%
% input: 
% ../gaze_data_with_em/run-1/sub-01_ses-movie_task-movie_run-1_converted.arff
% ../frame_motion/fg_av_ger_seg0.motion
% ex. ComputeRegressorsAll('eye_movement_type', 'regressors');

function ComputeRegressorsAll(attName, outDir)
    subIds = {'01'; '02'; '03'; '04'; '05'; '06'; '09'; '10'; '14'; '15'; '16'; '17'; '18'; '19'; '20'};
    saccShare = [0.1236; 0.0880; 0.0870; 0.1156; 0.0073; 0.1347; 0.0713; 0.1085; 0.0477; 0.1147; 0.0837; 0.0680; 0.1040; 0.0581; 0.0476];
    spShare = [ 0.1074; 0.1915; 0.1859; 0.1554; 0.0094; 0.1536; 0.1928; 0.1810; 0.1901; 0.1368; 0.1309; 0.1207; 0.1152; 0.1741; 0.1201];

	for subInd=1:size(subIds,1)
        regex = ['../gaze_data_with_em/run-*/sub-' subIds{subInd,1} '*_ses-movie_task-movie_run-*_converted.arff'];

        arffFiles = glob(regex);

        spExt = '_sp.txt';      % smooth pursuit
        saccExt = '_sacc.txt';  % saccade

        for i=1:size(arffFiles,1)
            arffFile = arffFiles{i,1};
            [dir, base, ext] = fileparts(arffFile);
            disp(arffFile)

            [start, duration, spValues, saccValues] = ComputeRegressors(arffFile, attName, saccShare(subInd), spShare(subInd));

            spSaveFile = [outDir '/' base spExt];
            dlmwrite(spSaveFile, [start duration spValues], 'precision', 10);
            saccSaveFile = [outDir '/' base saccExt];
            dlmwrite(saccSaveFile, [start duration saccValues], 'precision', 10);
        end

   end
end
