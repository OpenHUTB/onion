%% Comparing Whole Genomes
% This example shows how to compare whole genomes for organisms, which
% allows you to compare the organisms at a very different resolution
% relative to single gene comparisons. Instead of just focusing on the
% differences between homologous genes you can gain insight into the
% large-scale features of genomic evolution.

% 参考：Comparing Whole Genomes的例子
% https://ww2.mathworks.cn/help/bioinfo/ug/comparing-whole-genomes.html
% 需要引用数据"chlamydia.mat"和"sequences.mdb"


%%
% This example uses two strains of Chlamydia, Chlamydia trachomatis and
% Chlamydophila pneumoniae. These are closely related bacteria that cause
% different, though both very common, diseases in humans. Whole genomes are
% available in the GenBank(R) database for both organisms. 


%% Retrieving the Genomes
% You can download these genomes using the |getgenbank| function. First,
% download the Chlamydia trachomatis genome. Notice that the genome is
% circular and just over one million bp in length. These sequences are
% quite large so may take a while to download.
%
%   seqtrachomatis = getgenbank('NC_000117');

%%
% Next, download Chlamydophila pneumoniae. This genome is also circular and
% a little longer at 1.2 Mbp.
%
%   seqpneumoniae = getgenbank('NC_002179');

%%
% For your convenience, previously downloaded sequences are included in a
% MAT-file. Note that data in public repositories is frequently curated and 
% updated. Hence, the results of this example might be slightly different
% when you use up-to-date datasets.

data_dir = fullfile(matlabroot, 'examples', 'bioinfo', 'data');
load(fullfile(data_dir, 'chlamydia.mat'), 'seqtrachomatis', 'seqpneumoniae')

%% 
% A very simple approach for comparing the two genomes is to perform
% pairwise alignment between all genes in the genomes. Given that these are
% bacterial genomes, a simple approach would be to compare all ORFs in the
% two genomes. However, the GenBank data includes more information about
% the genes in the sequences. This is stored in the CDS field of the data
% structure. Chlamydia trachomatis has 895 coding regions, while
% Chlamydophila pneumoniae has 1112.

M = numel(seqtrachomatis.CDS)
N = numel(seqpneumoniae.CDS)

%%
% Most of the CDS records contain the translation to amino acid sequences.
% The first CDS record in the Chlamydia trachomatis data is a hypothetical
% protein of length 591 residues.
seqtrachomatis.CDS(1)

%%
% The fourth CDS record is for the gatA gene, which has product
% glutamyl-tRNA amidotransferase subunit A. The length of the product
% sequence is 491 residues.
seqtrachomatis.CDS(4)

%%
% A few of the Chlamydophila pneumoniae CDS have empty translations. Fill
% them in as follows. First, find all empty translations, then display the
% first empty translation.
missingPn = find(cellfun(@isempty,{seqpneumoniae.CDS.translation}));
seqpneumoniae.CDS(missingPn(1))

%%
% The function |featureparse| extracts features, such as the CDS, from the
% sequence structure. You can then use |cellfun| to apply |nt2aa| to the
% sequences with missing translations. 

allCDS = featureparse(seqpneumoniae,'Feature','CDS','Sequence',true); 
missingSeqs = cellfun(@nt2aa,{allCDS(missingPn).Sequence},'uniform',false); 
[seqpneumoniae.CDS(missingPn).translation] = deal(missingSeqs{:}); 
seqpneumoniae.CDS(missingPn(1)) 

%% Performing Gene Comparisons
% To compare the gatA gene in Chlamydia trachomatis with all the CDS genes
% in Chlamydophila pneumoniae, put a |for| loop around
% the |nwalign| function. You could alternatively use local alignment
% (|swalign|). 

tic
gatAScores = zeros(1,N);
for inner = 1:N
    gatAScores(inner) = nwalign(seqtrachomatis.CDS(4).translation,...
        seqpneumoniae.CDS(inner).translation);
end
toc % |tic| and |toc| are used to report how long the calculation takes.

%%
% A histogram of the scores shows a large number of negative scores and one
% very high positive score.
hist(gatAScores,100)
title(sprintf(['Alignment Scores for Chlamydia trachomatis %s\n',... 
    'with all CDS in Chlamydophila pneumoniae'],seqtrachomatis.CDS(4).gene))

%%
% As expected, the high scoring match is with the gatA gene in
% Chlamydophila pneumoniae.
[gatABest, gatABestIdx] = max(gatAScores);
seqpneumoniae.CDS(gatABestIdx)

%%
% The pairwise alignment of one gene from Chlamydia trachomatis with all
% genes from Chlamydophila pneumoniae takes just under a minute on an
% Intel(R) Pentium 4, 2.0 GHz machine running Windows(R) XP. To do this
% calculation for all 895 CDS in Chlamydia trachomatis would take about 12
% hours on the same machine. Uncomment the following code if you want to
% run the whole calculation.
%
%   scores = zeros(M,N);
%   parfor outer = 1:M
%      theScore = zeros(1,outer);
%      theSeq = seqtrachomatis.CDS(outer).translation;
%      for inner = 1:N
%          theScore(inner) = ...
%              nwalign(theSeq,...
%              seqpneumoniae.CDS(inner).translation);
%      end
%      scores(outer,:) = theScore;
%   end

%%
% Note the command |parfor| is used in the outer loop. If your machine is
% configured to run multiple _labs_ then the outer loop will be executed in
% parallel. For a full understanding of this construct, see |doc parfor|.


%% Investigating the Meaning of the Scores
% The distributions of the scores for several genes show a pattern. The
% CDS(3) of Chlamydia trachomatis is the gatC gene. This has a relatively
% short product,aspartyl/glutamyl-tRNA amidotransferase subunit C, with
% only 100 residues.

gatCScores = zeros(1,N);
for inner = 1:N
    gatCScores(inner) = nwalign(seqtrachomatis.CDS(3).translation,...
        seqpneumoniae.CDS(inner).translation);
end
figure
hist(gatCScores,100)
title(sprintf(['Alignment Scores for Chlamydia trachomatis %s\n',... 
    'with all CDS in Chlamydophila pneumoniae'],seqtrachomatis.CDS(3).gene))
xlabel('Score');ylabel('Number of genes');

%%
% The best score again corresponds to the same gene in the Chlamydophila
% pneumoniae.
[gatCBest, gatCBestIdx] = max(gatCScores);
seqpneumoniae.CDS(gatCBestIdx).product

%%
% CDS(339) of Chlamydia trachomatis is the uvrA gene. This has a very long
% product, excinuclease ABC subunit A, of length 1786.
uvrAScores = zeros(1,N);
for inner = 1:N
    uvrAScores(inner) = nwalign(seqtrachomatis.CDS(339).translation,...
        seqpneumoniae.CDS(inner).translation);
end
figure
hist(uvrAScores,100)
title(sprintf(['Alignment Scores for Chlamydia trachomatis %s\n',... 
    'with all CDS in Chlamydophila pneumoniae'],seqtrachomatis.CDS(339).gene))
xlabel('Score');ylabel('Number of genes');

[uvrABest, uvrABestIdx] = max(uvrAScores);
seqpneumoniae.CDS(uvrABestIdx)

%%
% The distribution of the scores is affected by the length of the
% sequences, with very long sequences potentially having much higher or
% lower scores than shorter sequences. You can normalize for this in a
% number of ways. One way is to divide by the length of the
% sequences. 

lnormgatABest = gatABest./length(seqtrachomatis.CDS(4).product)
lnormgatCBest = gatCBest./length(seqtrachomatis.CDS(3).product)
lnormuvrABest = uvrABest./length(seqtrachomatis.CDS(339).product)

%%
% An alternative normalization method is to use the self alignment score,
% that is the score from aligning the sequence with itself.

gatASelf = nwalign(seqtrachomatis.CDS(4).translation,...
    seqtrachomatis.CDS(4).translation);
gatCSelf = nwalign(seqtrachomatis.CDS(3).translation,...
    seqtrachomatis.CDS(3).translation);
uvrASelf = nwalign(seqtrachomatis.CDS(339).translation,...
    seqtrachomatis.CDS(339).translation);
normgatABest = gatABest./gatASelf
normgatCBest = gatCBest./gatCSelf
normuvrABest = uvrABest./uvrASelf


%% Using Sparse Matrices to Reduce Memory Usage
% The all-against-all alignment calculation not only takes a lot of time,
% it also generates a large matrix of scores. If you are looking for
% similar genes across species, then the scores that are interesting are
% the positive scores that indicate good alignment. However, most of these
% scores are negative, and the actual values are not particularly useful
% for this type of study. Sparse matrices allow you to store the
% interesting values in a more efficient way. 

%%
% The sparse matrix, |spScores|, in the MAT-file |chlamydia.mat| contains
% the positive values from the all against all pairwise alignment
% calculation normalized by self-alignment score.

load('chlamydia.mat','spScores')


%%
% With the matrix of scores you can look at the distribution of scores of
% Chlamydophila pneumoniae genes aligned with Chlamydia trachomatis and the
% converse of this, Chlamydia trachomatis genes aligned with Chlamydophila
% pneumoniae genes

figure
subplot(2,1,1)
hist(max(spScores),100)
title('Highest Alignment Scores for Chlamydophila pneumoniae Genes')
xlabel('Score');ylabel('Number of genes');
subplot(2,1,2)
hist(max(spScores,[],2),100)
title('Highest Alignment Scores for Chlamydia trachomatis Genes')
xlabel('Score');ylabel('Number of genes');

%%
% Remember that there are 1112 CDS in Chlamydophila pneumoniae and only 895
% in Chlamydia trachomatis. The high number of zero scores in the top
% histogram indicates that many of the extra CDS in Chlamydophila
% pneumoniae do not have good matches in Chlamydia trachomatis.

%%
% Another way to visualize the data is to look at the positions of points
% in the scores matrix that are positive. The sparse function |spy| is an
% easy way to quickly view dotplots of matrices. This shows some
% interesting structure in the positions of the high scoring matches.
figure
spy(spScores > 0)
title(sprintf('Dot Plot of High-Scoring Alignments.\nNormalized Threshold = 0'))
%%
% Raise the threshold a little higher to see clear diagonal lines in the
% plot.
spy(spScores >.1)
title(sprintf('Dot Plot of High-Scoring Alignments.\nNormalized Threshold = 0.1'))
%%
% Remember that these are circular genomes, and it seems that the starting
% points in GenBank are arbitrary. Permute the scores matrix so that
% the best match of the first CDS in Chlamydophila pneumoniae is in the
% first row to see a clear diagonal plot. This shows the synteny
% between the two organisms.

[bestScore bestMatch] = max(spScores(:,1));
spy(spScores([bestMatch:end 1:bestMatch-1],:)>.1);
title('Synteny Plot of Chlamydophila pneumoniae and Chlamydia trachomatis')

%% Looking for Homologous Genes
% Genes in different genomes that are related to each other are said to be
% homologous. Similarity can be by speciation (orthologous genes) or by
% replication (paralogous genes). Having the scoring matrix lets you look
% for both types of relationships.

%%
% The most obvious way to find orthologs is to look for the highest scoring
% pairing for each gene. If the score is significant then these best
% reciprocal pairs are very likely to be orthologous.

[bestScores, bestIndices] = max(spScores);

%%
% The variable bestIndices contains the index of the best reciprocal pairs
% for the genes in Chlamydophila pneumoniae. Sort the best scores and
% create a table to compare the description of the best reciprocal pairs
% and discover very high similarity between the highest scoring best
% reciprocal pairs.

[orderedScores, permScores] = sort(full(bestScores),'descend');
matches = [num2cell(orderedScores)',num2cell(bestIndices(permScores))',...
    num2cell((permScores))',...
    {seqtrachomatis.CDS(bestIndices(permScores)).product;...
    seqpneumoniae.CDS((permScores)).product; }'];

for count = 1:7
    fprintf(['Score %f\nChlamydia trachomatis Gene    : %s\n',...
        'Chlamydophila pneumoniae Gene : %s\n\n'],...
    matches{count,1}, matches{count,4}, matches{count,5})
end
%% 
% You can use the Variable Editor to look at the data in a spreadsheet format.
%%
% 
%   open('matches')
% 
%%
% Compare the descriptions to see that the majority of the best
% reciprocal pairs have identical descriptions. 
exactMatches = strcmpi(matches(:,4),matches(:,5));
sum(exactMatches)

%%
% Perhaps more interesting are the best reciprocal pairs where the
% descriptions are not identical. Some are simply differences in how the
% same gene is described, but others show quite different descriptions.

mismatches = matches(~exactMatches,:);
for count = 1:7
    fprintf(['Score %f\nChlamydia trachomatis Gene    : %s\n',...
        'Chlamydophila pneumoniae Gene : %s\n\n'],...
        mismatches{count,1}, mismatches{count,4}, mismatches{count,5})
end
%%
% View data for mismatches.
%
%%
% 
%   open('mismatches')
% 
%%
% Once you have the scoring matrix this opens up many possibilities for
% further investigation. For example, you could look for CDS where there
% are multiple high scoring reciprocal CDS. See Cristianini and Hahn [1]
% for further ideas.

%% References
% [1] Cristianini, N. and Hahn, M.W., "Introduction to Computational
%     Genomics: A Case Studies Approach", Cambridge University Press, 2007.
