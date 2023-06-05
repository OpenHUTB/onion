
rng(32);

Word = {'爵士乐';
    '古典乐';
    '嘻哈音乐';
    '迪斯科';
    '乡村音乐';
    '忧郁布鲁斯';
    '重金属音乐';
    '流行乐';
    '雷鬼乐';
    '摇滚乐'};
Count = [11;23;13;44;55;10;9;88;71;69];
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