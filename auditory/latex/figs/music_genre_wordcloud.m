Word = {'jazz';
    'classical';
    'hiphop';
    'disco';
    'country';
    'blues';
    'mental';
    'pop';
    'reggae';
    'rock'};


%% 神经网络预测结果的词云图
rng(98)

Count = [11;23;13;44;55;10;9;88;71;69];
tbl = table(Word, Count);

numWords = size(tbl,1);
colors = rand(numWords, 3);  

figure
wordcloud(tbl, 'Word', 'Count', 'Color', colors);
title("Sonnets Word Cloud")

%% 大脑预测的词云图

rng(77)

Count = [22;15;21;56;48;34;89;34;26;37];
tbl = table(Word, Count);

numWords = size(tbl,1);
colors = rand(numWords, 3);

figure
wordcloud(tbl, 'Word', 'Count', 'Color', colors);
title("Sonnets Word Cloud")


%%
% load sonnetsTable
% head(tbl)
% 
% figure
% wordcloud(tbl,'Word','Count');
% title("Sonnets Word Cloud")