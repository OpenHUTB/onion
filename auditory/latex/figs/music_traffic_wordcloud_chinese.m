
rng(32);

% https://www.iotword.com/2032.html
Word = {'空调声';
    '手提钻';
    '儿童玩耍声';
    '狗叫声';
    '钻孔声';
    '街道音乐';
    '枪声';
    '警笛声';
    '引擎空转声';
    '汽车鸣笛声'};
Count = [11;23;13;44;55;10;9;88;71;69];
tbl = table(Word, Count);

numWords = size(tbl,1);
colors = rand(numWords, 3);  

figure
wordcloud(tbl, 'Word', 'Count', 'Color', colors);
title("Sonnets Word Cloud")


%%
rng(2323);
Word = {'空调声';
    '手提钻';
    '儿童玩耍声';
    '狗叫声';
    '钻孔声';
    '街道音乐';
    '枪声';
    '汽车鸣笛声';
    '警笛声';
    '引擎空转声'};
Count = [11;23;13;44;55;10;9;88;71;69];
tbl = table(Word, Count);

numWords = size(tbl,1);
colors = rand(numWords, 3);  

figure
wordcloud(tbl, 'Word', 'Count', 'Color', colors);
title("Sonnets Word Cloud")