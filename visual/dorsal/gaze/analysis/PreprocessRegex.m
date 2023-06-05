% function PreprocessRegex.m
%
% 对所有受试者的视频段1到8进行预处理
% This function finds all the directories in the provided regular expression
% and preprocesses the data for runIds from 1 to 8
%
% input:
%   regex   -   dir regex
%
% ex. PreprocessRegex('/mnt/scratch/ioannis/studyforrest_preprocessed/sub-*/')
% 
% PreprocessRegex('/data2/whd/workspace/data/neuro/software/ds000113/sub-*/')

function PreprocessRegex(regex)
    init_env
    
    dirs = glob(regex);

    runIds = 1:8;

    for dir=dirs'
        % prepare data
        dst_dir = fullfile(dir{1}, 'ses-movie', 'func');
        if exist(dst_dir, 'dir')
           rmdir(dst_dir, 's'); 
        end
        path_names = strsplit(dir{1}, '/');
        cur_subject = path_names(end-1);
        src_dir = fullfile(raw_data_dir, cur_subject, 'ses-movie', 'func');
        copyfile(src_dir{1}, dst_dir);
        extract_all_file(dst_dir);
        
        for runId=runIds
            fprintf('Processing run %d from %s\n', runId, dir{1});
            Preprocess(dir{1}, runId);  
        end
    end
end

% '/data2/whd/workspace/data/neuro/software/ds000113/sub-01/ses-movie/func/meansub-01_ses-movie_task-movie_run-1_bold.nii'
% is empty

% ds000113/sub-01/anatomy/highres001.nii" does not exist.
% T1-weighted image
% Location sub<ID>/anatomy/highres001.nii.gz
% locate: ds000113/sub-01/ses-forrestgump/anat/sub-01_ses-forrestgump_T1w.nii
