% Export brain volume images
% 将.mat格式的文件 导出成 .nii 格式的大脑激活镜像
%
% This script requires BrainDecoderToolbox2 (https://github.com/KamitaniLab/BrainDecoderToolbox2).
%     *  Initial version.
%
% 需要下载：./data/Subject%0d_SpaceTemplate.nii

clear;

%% 目录设置
cur_dir = fileparts(mfilename('fullpath'));
run(fullfile(cur_dir, 'init.m'));

addpath(fullfile(matlabroot, 'software', 'matlab_utils', 'spm12'));

for i = 1:5
    bdata_file    = fullfile(workDir, sprintf( 'Subject%0d.mat', i));
    template_file = fullfile(workDir, 'preprocessed_fMRI_features', sprintf('Subject%0d_SpaceTemplate.nii', i));
    output_file   = fullfile(resultsDir, sprintf('Subject%0d_Func.nii', i));
    
    [dataset, metadata] = load_data(bdata_file);

    % Voxel data
    voxel_data = get_dataset(dataset, metadata, 'VoxelData');

    % Voxel xyz
    voxel_x = get_metadata(metadata, 'voxel_x', 'RemoveNan', true);
    voxel_y = get_metadata(metadata, 'voxel_y', 'RemoveNan', true);
    voxel_z = get_metadata(metadata, 'voxel_z', 'RemoveNan', true);

    % Exporting to .nii image (1st volume)
    export_volumeimage(output_file, voxel_data(1, :), [voxel_x; voxel_y; voxel_z], template_file);
end
