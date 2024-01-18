function rsa_searchlight_regressmodels()
    %% rsa_searchlight_regressmodels()
    %
    % computes rdms using a spherical searchlight approach and performs regression
    % on candidate model RDMs
    %
    % Paul Muhle-Karbe, 2023  
    
      params = rsa_searchlight_params();

      % group-level whole brain mask (generated with gen group mask script)
      gmaskMat = fmri_io_nifti2mat([params.names.groupMask '.nii'], params.dir.maskDir);
      gmaskVect = gmaskMat(:);
      gmaskIDsBrain = find(~isnan(gmaskVect));
      grpDir = [params.dir.inDir params.dir.subDir.GRP];
      helperpath = '/Volumes/Backup/fourrooms/functions/Helper';
      addpath(helperpath);
      
      load('disty2.mat');
      mrdm_comp = disty2.comp;
      mrdm_sep  = disty2.sep;
      mrdm_quad = disty2.quad;

      pred1 =[];pred2=[];pred3=[];
      
      for i = 1:length(mrdm_comp)
        pred1 = [pred1;mrdm_comp(i+1:end,i)];
        pred2 = [pred2;mrdm_sep(i+1:end,i)];
        pred3 = [pred3;mrdm_quad(i+1:end,i)];
      end
      
      nruns  = 2;
      nconds = 8;
      thresh = 0;

      voxel_selection = 1;
      whiten_method   = 1;
    
      subs = [301,302,304,305,306,307,308,309,310,311,312,314,315,316,317,318,319,320,323,324,325,326,327,328,329,330,331];
    
      modeldir = 'CurrentRoom_by_GoalState_byContext';
    
      for ii = 1:length(subs)
            warning('off','all');
            subID = num2str(subs(ii));
            
            % navigate to subject folder
            subStr = params.names.subjectDir(subID);
            subDir = [params.dir.inDir subStr '/'];
    
            disp(['Searchlight RSA Model Regressions - processing subject ' num2str(subs(ii))]);
            cd(['/Volumes/Backup/fourrooms/preprocessing/smooth_mni_6mm/sub-',num2str(subs(ii)),'/Models/',modeldir,'']);
            
            spmDir = (['/Volumes/Backup/fourrooms/preprocessing/smooth_mni_6mm/sub-',num2str(subs(ii)),'/Models/',modeldir]);
            rsaDir = (['/Volumes/Backup/fourrooms/preprocessing/smooth_mni_6mm/sub-',num2str(subs(ii)),'/Models/',modeldir]);
            
            % load SPM.mat
            cd(spmDir);
            SPM = load('SPM.mat');
            SPM = SPM.SPM;
            
    
            % import all betas
            bStruct = struct();
            [bStruct.b, bStruct.events] = rsa_helper_getBetas(SPM, params.num.runs, params.num.conditions, params.num.motionregs);
            bStruct.b = reshape(bStruct.b, [prod(size(gmaskMat)), size(bStruct.b, 4)]);
            bStruct.b = reshape(bStruct.b, [size(bStruct.b, 1), params.num.conditions, params.num.runs]);
            bStruct.events = reshape(bStruct.events, [params.num.conditions, params.num.runs]);
            
            %Select conditions of interest
            bStruct.b = bStruct.b(:,[2:2:16],:);
            bStruct.events = bStruct.events([2:2:16],:);
    
            X = [pred1,pred2,pred3];
            
            rdmSet = []; 
            disp('Loading Residuals...')
            Resids = spm_data_read(SPM.xY.VY,gmaskIDsBrain);
            disp('Residuals Loaded!')
            corrs = [];

            %Loop over searchlights
            for sphereID = 1:length(gmaskIDsBrain)
                disp(['Searchlight number ',num2str(sphereID),'']);
                % obtain coordinates of centroid
                [x, y, z] = ind2sub(size(gmaskMat), gmaskIDsBrain(sphereID));
                % create spherical mask
                [sphereIDs, ~] = fmri_mask_genSphericalMask([x, y, z], params.rsa.radius, gmaskMat);
                
                %Whiten voxels within sphere
                idces = fmri_mask_genSphericalMask([x, y, z], params.rsa.radius, gmaskMat);
               
                for ivox = 1:length(idces)
                    ind(ivox) = find(gmaskIDsBrain == idces(ivox));
                end
    
                if whiten_method == 1
                    Y = Resids(:,ind);
                    [u_hat,resMS,Sw_hat,beta_hat,shrinkage,trRR] = rsa.spm.noiseNormalizeBeta(Y,SPM);
                    %%%Select relevant cnditions
                    p        = u_hat([2:2:16],:)';%4 rooms 8 2 contexts in run 2
                    p(:,:,2) = u_hat([33:2:47],:)';%run2
                    betas = permute(p, [2, 3, 1]);
                    
                elseif whiten_method == 2
                    betas = squeeze(bStruct.b(sphereIDs, :, :));
                    betas = permute(betas, [2, 3, 1]);
                    Y = spm_data_read(SPM.xY.VY,idces);
                    % weight data and remove filter confounds (?)
                    KWY = spm_filter(SPM.xX.K,SPM.xX.W*Y);
                    % compute residuals
                    r = spm_sp('r',SPM.xX.xKXs,KWY);
                    for runID =1:size(betas,2)
                      betas(:,runID,:) = squeeze(betas(:,runID,:)) * mpower(rsa.stat.covdiag(squeeze(r(:,runID,:))),-.5);
                    end
                    
                elseif whiten_method == 3
                    betas = squeeze(bStruct.b(sphereIDs, :, :));
                    betas = permute(betas, [2, 3, 1]);
                    
                end
               
                %mask betas with sphere, only use masked values
                %Remove NaN voxels from the searchlight
                if length(find(isnan(betas))) > 0
                    betas(isnan(betas)) = [];
                    betas = reshape(betas,[nconds,nruns,(length(betas)/(nconds*nruns))]);
                end
                
                
                if voxel_selection == 1
                    r1_odd  = [squeeze(betas(1,1,:))];
                    r2_odd  = [squeeze(betas(2,1,:))];
                    r3_odd  = [squeeze(betas(3,1,:))];
                    r4_odd  = [squeeze(betas(4,1,:))];
                    r5_odd  = [squeeze(betas(5,1,:))];
                    r6_odd  = [squeeze(betas(6,1,:))];
                    r7_odd  = [squeeze(betas(7,1,:))];
                    r8_odd  = [squeeze(betas(8,1,:))];
                    
                    r1_even  = [squeeze(betas(1,2,:))];
                    r2_even  = [squeeze(betas(2,2,:))];
                    r3_even  = [squeeze(betas(3,2,:))];
                    r4_even  = [squeeze(betas(4,2,:))];
                    r5_even  = [squeeze(betas(5,2,:))];
                    r6_even  = [squeeze(betas(6,2,:))];
                    r7_even  = [squeeze(betas(7,2,:))];
                    r8_even  = [squeeze(betas(8,2,:))];
    
                    
                    betas_odd  = [r1_odd,r2_odd,r3_odd,r4_odd,r5_odd,r6_odd,r7_odd,r8_odd];%
                    betas_even = [r1_even,r2_even,r3_even,r4_even,r5_even,r6_even,r7_even,r8_even];%
                    
                    [rho]      = corr(betas_odd',betas_even','Type','Speaman');
                    c          = diag(rho);
                    outlier    = find((c<0));
                    nonoutlier = find((c>0));
                    
               end
            
           rdm       = rsa_compute_rdmSet_cval(betas, params.rsa.metric);
           rdm_cv    = rdm(1:length(rdm)/2,(length(rdm)/2)+1:length(rdm));
           
           comp      = mrdm_comp .* rdm_cv;
           bcomp     = sum(comp(:));
           sep       = mrdm_sep .* rdm_cv;
           bsep      = sum(sep(:)); 
           quad      = mrdm_quad .* rdm_cv;
           bquad     = sum(quad(:));
           
           betas_glm = [bcomp;bsep;bquad];
           corrs     = [corrs,betas_glm]; 
            
             
           [~, rdmSet(sphereID, :, :)]        = rsa_compute_averageCvalRDMs(rdm, nruns,nconds);
           
           %Display progress of the searchlights being completed
           if rem(sphereID,500) == 0
                disp(['',num2str(sphereID),' out of ',num2str(length(gmaskIDsBrain)),' searchlights completed'])
           end
            
        end
        
        
        results        = struct();
        results.betas  = corrs;
        results.params = params;
        
        if ~exist('rsaDir', 'dir')
            mkdir(rsaDir);
        end

        cd(rsaDir);

        % save results
        subRDM           = struct();
        subRDM.rdms      = rdmSet;
        subRDM.events    = bStruct.events(:, 1);
        subRDM.subID     = subID;
        subRDM.indices   = gmaskIDsBrain;
        
        save('subRDM_glm_CSQ_GOAL_NDR','subRDM');        

        if ~exist('params.dir.subDir.rsabetas', 'dir')
            mkdir(params.dir.subDir.rsabetas);
        end

        cd(params.dir.subDir.rsabetas);
        save('results_glm_CSQ_GOAL_NDR','results');
        
    end

end