%% 为人科物种建立系统发育树
% 此示例说明如何从人科类群（也称为猩猩科）的线粒体DNA(mtDNA)序列构建系统发育树。
% 这个家庭包括大猩猩、黑猩猩、猩猩和人类。

% 参考：https://ww2.mathworks.cn/help/bioinfo/ug/building-a-phylogenetic-tree-for-the-hominidae-species.html
% 需要加载数据文件：mitochondria.mat、primates.mat、sequences.mdb
data_dir = fullfile(matlabroot, 'examples', 'bioinfo');
addpath(data_dir);

%% 介绍
% 线粒体 D环 是动物 DNA 中突变最快的序列区域之一，因此常用于比较密切相关的生物。
% 现代人的起源是一个备受争议的问题，已通过使用 mtDNA 序列得到解决。
% 人类 mtDNA 的有限遗传变异性已根据最近的共同遗传祖先得到解释，
% 因此暗示所有现代人口的 mtDNA 可能起源于生活在非洲不到 200,000 年的单个女性。

%% 从 GenBank® 中检索序列数据
% 此示例使用为具有以下 GenBank 登录号的不同人科物种分离的线粒体 D 环序列。

%        物种描述                  GenBank登录号
data = {'German_Neanderthal'      'AF011222';  % 德国尼安德特人
        'Russian_Neanderthal'     'AF254446';  % 俄罗斯尼安德特人
        'European_Human'          'X90314'  ;  % 欧洲人
        'Mountain_Gorilla_Rwanda' 'AF089820';  % 卢旺达山地大猩猩
        'Chimp_Troglodytes'       'AF176766';  % 黑猩猩穴居人
        'Puti_Orangutan'          'AF451972';  % 菩提猩猩
        'Jari_Orangutan'          'AF451964';  % ?
        'Western_Lowland_Gorilla' 'AY079510';  % 西非低地大猩猩
        'Eastern_Lowland_Gorilla' 'AF050738';  % 东非低地大猩猩
        'Chimp_Schweinfurthii'    'AF176722';  % 黑猩猩Schweinfurthii ?
        'Chimp_Vellerosus'        'AF315498';
        'Chimp_Verus'             'AF176731';
       };
   
%%
% 可以使用循环内的函数 |getgenbankfor| 从 NCBI 数据库中检索序列并将它们加载到 MATLAB® 中。
%
%   for ind = 1:length(data)       
%       primates(ind).Header   = data{ind,1};
%       primates(ind).Sequence = getgenbank(data{ind,2},'sequenceonly','true');
%   end

%%

% 为方便起见，以前下载的序列包含在 MAT 文件中。
% 请注意，公共存储库中的数据经常被整理和更新；
% 因此，当您使用最新序列时，此示例的结果可能会略有不同。

load('primates.mat')

%% 使用距离方法构建 UPGMA 系统发育树
% 使用“Jukes-Cantor”公式和使用“UPGMA”距离方法的系统发育树计算成对距离。
% 由于序列未预先对齐，因此在计算距离之前使用 |seqpdist| 执行成对对齐。
% 参考：https://github.com/Ming-Lian/Memo/blob/master/%E6%9E%84%E5%BB%BA%E7%B3%BB%E7%BB%9F%E8%BF%9B%E5%8C%96%E6%A0%91%EF%BC%9A%E4%BB%8E%E5%8E%9F%E7%90%86%E5%88%B0%E6%93%8D%E4%BD%9C.md

distances = seqpdist(primates,'Method','Jukes-Cantor','Alpha','DNA');
UPGMAtree = seqlinkage(distances,'UPGMA',primates)

h = plot(UPGMAtree,'orient','top');
title('UPGMA Distance Tree of Primates using Jukes-Cantor model');
ylabel('Evolutionary distance')

%% 使用距离方法构建邻接系统发育树
% 在分析物种之间的同源序列时，交替树拓扑结构很重要。
% 可以使用 |seqneighjoin| 函数构建邻接树。
% 邻接树使用上面计算的成对距离来构建树。
% 该方法使用最小进化方法执行聚类。

NJtree = seqneighjoin(distances,'equivar',primates)

h = plot(NJtree,'orient','top');
title('Neighbor-Joining Distance Tree of Primates using Jukes-Cantor model');
ylabel('Evolutionary distance')

%% 比较树拓扑
% 请注意，不同的系统发育重建方法会导致不同的树拓扑。
% 邻接树将 Chimp Vellerosus 分组在与大猩猩的进化枝中，
% 而 UPGMA 树将它分组在黑猩猩和猩猩附近。
% |getcanonical| 函数可用于比较这些同构树。

sametree = isequal(getcanonical(UPGMAtree), getcanonical(NJtree))

%% 探索 UPGMA 系统发育树
% 可以通过考虑距"欧洲人类"条目的给定共祖距离内的节点（叶子和分支）来探索系统发育树，
% 并通过修剪掉不相关的节点将树减少到感兴趣的子分支。

names = get(UPGMAtree,'LeafNames')
[h_all,h_leaves] = select(UPGMAtree,'reference',3,'criteria','distance','threshold',0.3);

subtree_names = names(h_leaves)
leaves_to_prune = ~h_leaves;

pruned_tree = prune(UPGMAtree,leaves_to_prune)
h = plot(pruned_tree,'orient','top');
title('Pruned UPGMA Distance Tree of Primates using Jukes-Cantor model');
ylabel('Evolutionary distance')

%%
% 您可以使用 |view| 交互式工具进一步探索/编辑系统发育树。另见 |phytreeviewer|。

view(UPGMAtree,h_leaves)

%% 参考文献
% [1] Ovchinnikov, I.V., et al., "Molecular analysis of Neanderthal DNA
%     from the northern Caucasus", Nature, 404(6777):490-3, 2000.
%
% [2] Sajantila, A., et al., "Genes and languages in Europe: an analysis of
%     mitochondrial lineages", Genome Research, 5(1):42-52, 1995.
%
% [3] Krings, M., et al., "Neandertal DNA sequences and the origin of
%     modern humans", Cell, 90(1):19-30, 1997.
%
% [4] Jensen-Seaman, M.I. and Kidd, K.K., "Mitochondrial DNA variation and
%     biogeography of eastern gorillas", Molecular Ecology, 10(9):2241-7,
%     2001.
