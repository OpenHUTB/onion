% ComputeRegressorsAll.m
% 
% This function calls the ComputeRegressors for all the files in the regular
% expression with the correct motion file and saves the regressors in files with
% _sp.txt, _fix.txt, _motion.txt endings
% 对所有的文件调用ComputeRegressors脚本，在2秒间隔内得到一个各类型眼动有代表性的值，用_sp.txt, _fix.txt,
% _motion.txt文件保存在regressors文件夹中
%
% input:
%   regex   - ARFF格式文件的正则表达式 regular expresion to .arff files
%   attName - 用于间隔抽取的属性名 attribute name to use for intervals extraction 
%{
    {'time'                    }    {'INTEGER'}
    {'x'                       }    {'NUMERIC'}
    {'y'                       }    {'NUMERIC'}
    {'confidence'              }    {'NUMERIC'}
    {'frame_id'                }    {'INTEGER'}
    {'eye_movement_type'       }    {'INTEGER'}
    {'sacc_interval_index'     }    {'INTEGER'}
    {'intersacc_interval_index'}    {'INTEGER'}
    {'cluster_id'              }    {'NUMERIC'}
%}
%   outDir  - dir to save blocks
%
% ex. ComputeRegressorsAll('eye_movement_type', 'regressors');
%
% input: 
% ../gaze_data_with_em/run-1/sub-01_ses-movie_task-movie_run-1_converted.arff
% ../frame_motion/fg_av_ger_seg0.motion
% 
% 运行：
% ComputeRegressorsAll('eye_movement_type', 'regressors');

function ComputeRegressorsAll(attName, outDir)
    run('../init_env.m');
    if ~exist(outDir)
        mkdir(outDir);
    end

    subIds = {'01'; '02'; '03'; '04'; '05'; '06'; '09'; '10'; '14'; '15'; '16'; '17'; '18'; '19'; '20'};
    saccShare = [0.1236; 0.0880; 0.0870; 0.1156; 0.0073; 0.1347; 0.0713; 0.1085; 0.0477; 0.1147; 0.0837; 0.0680; 0.1040; 0.0581; 0.0476];  % 对应受试者的眼跳百分比
    spShare = [ 0.1074; 0.1915; 0.1859; 0.1554; 0.0094; 0.1536; 0.1928; 0.1810; 0.1901; 0.1368; 0.1309; 0.1207; 0.1152; 0.1741; 0.1201];

	for subInd=1:size(subIds,1)
        % run_detection.py的输出，保存在目录gaze_data_with_em中
        regex = ['../gaze_data_with_em/run-*/sub-' subIds{subInd,1} '*_ses-movie_task-movie_run-*_converted.arff'];

        arffFiles = glob(regex);

        spExt = '_sp.txt';         % smooth pursuit
        saccExt = '_sacc.txt';     % saccade
        motionExt = '_motion.txt'; % motion

        for i=1:size(arffFiles,1)
            arffFile = arffFiles{i,1};
            [dir, base, ext] = fileparts(arffFile);
            disp(arffFile)
            pos = findstr(arffFile, 'run-');
            runNum = str2num(arffFile(pos(1)+4)) - 1;
            motionFile = ['../frame_motion/fg_av_ger_seg' num2str(runNum) '.motion'];  % 8个片段的运动（不包括对应的多个人了）
            disp(motionFile);
            disp('----------------');

            [start, duration, spValues, saccValues, motionValues] = ComputeRegressors(arffFile, attName, saccShare(subInd), spShare(subInd), motionFile);

            spSaveFile = [outDir '/' base spExt];
            dlmwrite(spSaveFile, [start duration spValues], 'precision', 10);
            saccSaveFile = [outDir '/' base saccExt];
            dlmwrite(saccSaveFile, [start duration saccValues], 'precision', 10);
            motionSaveFile = [outDir '/' base motionExt];
            dlmwrite(motionSaveFile, [start duration motionValues], 'precision', 10);
        end

   end
end
