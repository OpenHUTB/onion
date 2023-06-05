function Features = MakeFeatureSpace_MTF(Filename,Dir, FileLength, TR, MTFCode)
% Output is N (features) * T (FileLength/TR) matrix
% Filename = ['***.wav'];
% Dir... directry where the wave file is stored
% FileLength in second (e.g., 15)
% TR in second (e.g., 2)
% Model based on Chi et al. (2005), JASA, 118(2), 887-906.
%
% We recommend to use MTFCode = 24 (2018/5/5)
% You can modify parameter in MTFparams.m
%
%Requirement: gammatonegram (https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/)
%

% Load parameter
Params = MTFparams(MTFCode);

disp(['Making MTF features for ' Filename])

% Generate Cochleogram
[Coch, LogScale] = MakeCochleogram(Filename, Dir, Params,  0);

% Normalization
M = mean(reshape(Coch, [], 1));
SD = std(reshape(Coch, [], 1));
sCoch = (Coch - M ) / SD;      

% Feature extraction
Features = ExtractMTFfeatures(sCoch, LogScale, FileLength, TR, MTFCode);



