
Method = 'TaskType';

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

gtResp = [];
for ss  = 1:6
    ID = ['sub-0' num2str(ss)];

    % Load target voxels in the cerebral cortex
    load([DataDir 'VsetInfo_' ID '.mat'],'vset_info');
    ROI = vset_info.IDs;
    vset = ROI(1);
    voxelSetForm = [DataDir 'vset_' ID '/vset_%03d.mat'];
    load(sprintf(voxelSetForm,vset),'tvoxels');
    datasize = [96 96 72];

    % Load response data (Matrix of Time points * Voxels)
    load([DataDir 'RespData_' ID '.mat']);
    respTrn_ROI = RespData.respTrn;

    % Load task names
    load([DataDir 'TaskList.mat'],'TaskList');

    % Load stimulus information (Matrix of Time points * features)
    load([DataDir 'Stim_' Method '_' ID '.mat']);
    stimTrn = StimData.stimTrn;

    StimTrn = [];
    for ii = 1:size(stimTrn,1)
        if ~isempty(find(stimTrn(ii,:)))
            StimTrn(ii) = find(stimTrn(ii,:));
        else
            StimTrn(ii) = 0;
        end
    end

    % Correction for bold time delay
    RespTrn = cat(3,circshift(respTrn_ROI,-1), circshift(respTrn_ROI,-2), circshift(respTrn_ROI,-3));
    RespTrn = squeeze(mean(RespTrn,3));

    % Average for each task type
    tResp = [];
    NTask = 8; %Number of trials in the training dataset
    for tt = 1:103
        tdata = RespTrn( find(StimTrn==tt) , :);
        ttdata = reshape(tdata,size(tdata,1)/NTask,NTask,size(tdata,2));
        tResp = cat(1,tResp,nanmean(ttdata , 1) );
    end
    ttResp = tResp;
    % Average stimuli in the same task
    tResp = squeeze( nanmean(ttResp,2) );
    gtResp{ss} = tResp;
end


% Make dendrogram using other five subjects
for ss = 1:6
    ID = ['sub-0' num2str(ss)];
    gtResp_tmp = [];
    for jj = setdiff(1:6,ss)
    gtResp_tmp = [gtResp_tmp, gtResp{jj}];
    end

    % Hierarchical clustering analysis with dissimilarity matrix
    X = pdist(gtResp_tmp,'correlation');
    Y = linkage(X, 'average');

    N_Task = 103;
    N_Feature = size(Y,1); %Number of non-terminal node in dendrogram
    Trans = zeros(N_Task,N_Feature);
    for cc = 1:size(Y,1)

        for tt = 1:2
            if Y(cc,tt) <= N_Task
                Trans(Y(cc,tt), cc) = 1;
            else
                Trans(:,cc) = Trans(:,cc) + Trans(:,Y(cc,tt)- N_Task);
            end
        end
    end

    Trn = stimTrn*Trans;
    Val = stimVal*Trans;

    Stim.Trn = Trn;
    Stim.Val = Val;
    save([SaveDir 'HCABased_TaskType_' ID '.mat'],'Stim')
end
