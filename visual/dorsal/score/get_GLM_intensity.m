%% 从GLM得到的脑区中得到强度值

%%
addpath('utils');
proj_home = fileparts( fileparts( mfilename('fullpath') ) );
res_dir = fullfile(proj_home, 'result');
addpath(fullfile(proj_home, 'utils/', 'spm12/'));

%% Load Image Data
image_filename = '/data2/whd/workspace/sot/hart/gaze/analysis/sp_saccade_motion_analysis/sp_sacc_motion_analysis/sub-01/spmT_0001.nii';

[imageFileName, M, DIM, TF, df, mni, intensity] ...
    = get_image_file(image_filename);
mni = mni{1};
intensity = intensity{1};

%% Load Brain Area Data
X = load('data/TDdatabase');
wholeMaskMNIAll = X.wholeMaskMNIAll;


%% Filter with brain area
V1_mask = wholeMaskMNIAll.brodmann_area_17;
MT_mask = wholeMaskMNIAll.brodmann_area_21;
FEF_mask = wholeMaskMNIAll.brodmann_area_8;

% pcshow(pointCloud(V1_mask))
% pcshow(pointCloud(mni))


V1_intensity = zeros(1, length(V1_mask));
MT_intensity = zeros(1, length(MT_mask));
FEF_intensity = zeros(1, length(FEF_mask));
V1_idx = 0;
MT_idx = 0;
FEF_idx = 0;
for j = 1 : length(V1_mask)
    for i = 1 : length(mni)
        cur_mni = mni(i, :);
        if sum(cur_mni == V1_mask(j, :)) == 3  % coordinate match
            V1_idx = V1_idx + 1;
            V1_intensity(V1_idx) = intensity(i);
            break;
        end
    end
    disp(j)
end
V1_intensity = V1_intensity(1:V1_idx);  % crop 

for j = 1 : length(MT_mask)
    for i = 1 : length(mni)
        cur_mni = mni(i, :);
        if sum(cur_mni == MT_mask(j, :)) == 3
            MT_idx = MT_idx + 1;
            MT_intensity(MT_idx) = intensity(i);
            break;
        end
    end
    disp(j);
end
MT_intensity = MT_intensity(1:MT_idx);

for j = 1 : length(FEF_mask)
    for i = 1 : length(mni)
        cur_mni = mni(i, :);
        if sum(cur_mni == FEF_mask(j, :)) == 3
            FEF_idx = FEF_idx + 1;
            FEF_intensity(FEF_idx) = intensity(i);
            break;
        end
    end
    disp(j);
end
FEF_intensity = FEF_intensity(1:FEF_idx);

fprintf('V1 point number: %d\n', sum(V1_intensity ~= 0));  % 25
fprintf('MT point number: %d\n', sum(MT_intensity ~= 0));
fprintf('FEF point number: %d\n', sum(FEF_intensity ~= 0));

% V1 point number: 25
% MT point number: 70
% FEF point number: 70

%% Save Result
save(fullfile(res_dir, 'activation', 'V1_intensity'), 'V1_intensity', '-ascii');
save(fullfile(res_dir, 'activation', 'MT_intensity'), 'MT_intensity', '-ascii');
save(fullfile(res_dir, 'activation', 'FEF_intensity'), 'FEF_intensity', '-ascii');

