function [predictor,response,segmentsPerFile] = helperAudioPreprocess(ads,overlap)

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