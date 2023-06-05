
rng(32);

% Urban Sound 8K 
% https://cloud.tencent.com/developer/article/1655282?from=15425&areaSource=102001.1&traceId=MkrJYjuS0nKCwANxdkWMJ
% 副驾驶 shotgun：防止盗贼
Word = {'发动机空转';
    '手提钻';
    '儿童游戏';
    '狗吠';
    '空调';
    '钻探';
    '汽车加速';
    '汽车喇叭';
    '警笛';
    '街头音乐'};
Count = [11;23;13;44;55;10;9;88;71;69];
tbl = table(Word, Count);

numWords = size(tbl,1);
colors = rand(numWords, 3);  

figure
wordcloud(tbl, 'Word', 'Count', 'Color', colors);
title("Sonnets Word Cloud")


%%
rng(100);

Word = {'街头音乐';
    '手提钻';
    '儿童游戏';
    '狗吠';
    '空调';
    '钻探';
    '汽车加速';
    '警笛';
    '汽车喇叭';
    '发动机空转'};
Count = [11;23;13;44;55;10;9;88;71;69];
tbl = table(Word, Count);

numWords = size(tbl,1);
colors = rand(numWords, 3);  

figure
wordcloud(tbl, 'Word', 'Count', 'Color', colors);
title("Sonnets Word Cloud")