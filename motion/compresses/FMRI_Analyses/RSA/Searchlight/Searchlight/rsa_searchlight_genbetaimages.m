function rsa_searchlight_genbetaimages()
    %
    % generates grouplevel-masked volumes of regression coefficients
    % (to include only voxels that have values for all participants)
    %
    % Paul Muhle-Karbe, 2023
    
    params = rsa_searchlight_params();
    % load grouplevel mask
    gmaskMat = fmri_io_nifti2mat([params.names.groupMask '.nii'], params.dir.maskDir);
    gmaskVect = gmaskMat(:);

    allBetas = [];
    % generate images
    params = rsa_searchlight_params();
    
    subs = [301,302,304,305,306,307,308,309,310,311,312,314,315,316,317,318,319,320,323,324,325,326,327,328,329,330,331];

    for ii = 1:length(subs)
        subID = subs(ii);%params.num.goodSubjects(ii);
        disp(['processing subject ' num2str(subID)])
        % load mask indices
        subStr = sprintf('Sub%d', subID)

        maskIDs = load(['/Volumes/Backup/fourrooms/preprocessing/smooth_mni_6mm/sub-',num2str(subs(ii)),'/Models/Room_Move_Context_Corr_LowRes_Oct02_2022/subRDM_glm_context_UT_NoMinorDiag'])
        maskIDs = maskIDs.subRDM.indices;
        load(['/Volumes/Backup/fourrooms/preprocessing/smooth_mni_6mm/sub-',num2str(subs(ii)),'/Models/Room_Move_Context_Corr_LowRes_Oct02_2022/betas/results_glm_context_UT_NoMinorDiag'])
        betas = results.betas;
        
        % loop through models
        for modID = 1:size(betas, 1)
            % get mask indices of single subject betas
            [x, y, z] = ind2sub(size(gmaskMat), maskIDs);
            XYZ = [x y z]';
            % generate volume from indices and tau values
            volMat = fmri_volume_genVolume(size(gmaskMat), XYZ, betas(modID, :));
            % apply group-level mask
            volMat(isnan(gmaskVect)) = NaN;
            % add to big matrix
            allBetas(ii, modID, :, :, :) = volMat;

            % write volume
            fName = fullfile(params.dir.outDir, ['Searchlight_Context_UT_NoMinorDiag_' params.names.modelset '_mod' num2str(modID) '_sub' num2str(subID) '.nii']);
            fmri_io_mat2nifti(volMat, fName, 'rdm model betas (regression)', 16);
        end

    end

end