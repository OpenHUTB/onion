% function ContratAnalysis:
% this function creates contrast of SP vs fix. It uses te SPM.mat file within the provided directory
% 使用所提供目录下的SPM.mat文件创建平稳跟踪和注视的对比
% SPM manual: 42.7.8 Setting up contrasts

%% 构造Contrast：对感兴趣的解释变量进行比较
% T检验：构造Contrast向量
% F检验：构造Contrast矩阵
% 实验设计 -> 感兴趣effect -> contrast （所以contrast在数据采集之前就定下了）

% Bonferroni（邦费罗尼）校正的思想及其在fMRI数据分析中的问题
% Bonferroni校正的假设：p_{voxel} = p_{overall} / N，其中N为独立观测个数
% 这样做的理由是基于这样一个事实：在同一数据集上进行多个假设的检验，每20个假设中就有一个可能纯粹由于概率，而达到0.05的显著水平。
% 相邻体素的BOLD信号会相互独立吗？
%    头动等噪声对同一脑区的影响很相似
%    BOLD信号本身就对应着一定空间范围
%    预处理中的平滑

% SPM中的多重比较校正的原理
% 根据数据的空间相关程度计算独立观测个数（独立比较的次数N_{indepentent}
% 根据整体虚警概率p_{overall}和N_{indepentent}得到单个体素的p_{voxel}值
% p_{voxel} = p_{overall} / N_{indepentent}

%% 建立不同人之间的可比性：Normalization, ROI
% 多个被试者的统计分析：Fixed-effects Model、Random-effects Model

% 固定效应模型
% 假设：对于每个受试者，实验操作（观看视频）都有相同的效果。
% 使用所有受试者的数据构建统计测试
%     在进行t-检验之前 平局/连接不同受试者
% 对从单个受试者极端结果敏感
%     当其他受试者表现出较弱或者没有效应时，在一个受试者中有强烈的效应就会导致巨大的事件。
% 对受试者样本允许推断
%     在你自己的受试者组里可以说效果明显，但是不能泛化到其他你没有测试其他受试者中。

% 随机效应分析
% 假设：在群体之间这个效应是变化的
% 对受试者内部的分析进行解释
% 允许从绘制的受试者群体中进行推断
% 对于组比较特别重要
% 需要许多评论者和记录


%% input:
%   directory   - 搜索SPM.mat文件的目录. directory to search for SPM.mat file
% 
% 输出：
%   SPM.mat（包含传递给spm_spm.m的配置参数、设计矩阵、模型参数的估计，输出分析结果）
%   con_0061.nii   (70-)
%   spmT_0061.nii

function ContrastAnalysis(directory)
    % create batch file
    matlabbatch = [];
    matlabbatch{1}.spm.stats.con.spmmat = cellstr([directory '/SPM.mat']);

    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'SP > sacc';  % 平稳跟踪大于眼跳的对比
    contrast = zeros(8,24);  % 一个对比矩阵对应一个统计假设
    contrast(:,4) = 1;       % 想关注第4个变量，（排除其他自变量的影响下）第4个变量和因变量的关系
    contrast(:,16) = -1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.convec = contrast(:);
    % equivalent to                                               4(SP)       motion(10)  sacc(16)   
%    matlabbatch{1}.spm.stats.con.consess{1}.tcon.convec = [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ...
%                                                    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 2
%                                                    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 3
%                                                    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 4
%                                                    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 5
%                                                    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 6
%                                                    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 7
%                                                    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ... % session 8
%                                                    ];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'sacc > SP';
    contrast = zeros(8,24);
    contrast(:,4) = -1;
    contrast(:,16) = 1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'motion > sacc';
    contrast = zeros(8,24);
    contrast(:,10) = 1;
    contrast(:,16) = -1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'sacc > motion';
    contrast = zeros(8,24);
    contrast(:,10) = -1;
    contrast(:,16) = 1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'sp > motion';
    contrast = zeros(8,24);
    contrast(:,4) = 1;
    contrast(:,10) = -1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'motion > sp';
    contrast = zeros(8,24);
    contrast(:,4) = -1;
    contrast(:,10) = 1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'sp + motion > sacc';
    contrast = zeros(8,24);
    contrast(:,4) = 1;
    contrast(:,10) = 1;
    contrast(:,16) = -2;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{8}.tcon.name = 'SP > baseline';
    contrast = zeros(8,24);
    contrast(:,4) = 1;      % 只设置一个表示和baseline对比
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{8}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{8}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{9}.tcon.name = 'sacc > baseline';
    contrast = zeros(8,24);
    contrast(:,16) = 1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{9}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{9}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{10}.tcon.name = 'motion > baseline';
    contrast = zeros(8,24);
    contrast(:,10) = 1;
    contrast = contrast';
    matlabbatch{1}.spm.stats.con.consess{10}.tcon.convec = contrast(:);
    matlabbatch{1}.spm.stats.con.consess{10}.tcon.sessrep = 'none';

    
    % execute job
    spm_jobman('run',matlabbatch);
end
