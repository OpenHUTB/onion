function [predictor,response,segmentsPerFile] = helperAudioPreprocess(ads, overlap)
% 支持函数 helperAudioPreprocess 将 audioDatastore 对象和 log mel 频谱图之间的重叠百分比作为输入，
% 并返回适合输入到 VGGish 网络的预测变量和响应矩阵。

numFiles = numel(ads.Files);

% Extract predictors and responses for each file
for ii = 1:numFiles
    [audioIn,info] = read(ads);

    fs = info.SampleRate;
    features = vggishPreprocess(audioIn,fs,OverlapPercentage=overlap); 
    numSpectrograms = size(features,4);

    predictor{ii} = features;
    response{ii} = repelem(info.Label,numSpectrograms);
    segmentsPerFile(ii) = numSpectrograms;

end

% Concatenate predictors and responses into arrays
predictor = cat(4,predictor{:});
response = cat(2,response{:});
end