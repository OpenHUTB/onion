function [Coch, LogScale] =  MakeCochleogram(filename, Dir, Params, ImageSwitch)
%Coch...Cohleogram N (frequency bands) * T (Time) matrix 
%LogScale... Frequency bin between each freq. band
%
%filename ... wave format
%Params...STRFparams
%ImageSwitch... if 1, show spectrogram image
%
%Requirement: gammatonegram (https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/)

if ~exist('ImageSwitch')
    ImageSwitch = 1;
end


Nchannel = Params.Nchannel;
LowestFreq = Params.LowestFreq;
HighestFreq = Params.HighestFreq;


[Data, Fs] = audioread([Dir filename]);

% Transform to monaural
if size(Data,2) > 1
	Data(:,2) = []; 
end

% Resampling to 2*HighestFreq 
rsData = resample(Data,2*HighestFreq,Fs);
rsFs = 2*HighestFreq;
 
if Params.cNorm
    %Normalization to mean 0, sd 1
    rsData = zscore(rsData);
end

% Gammatone gram
% Note that output CentFreq array is in the reverse order (CentFreq(1) i the
% Highest frequency), but Coch matrix rows are in the normal order.
[Coch,CentFreq] = gammatonegram(rsData,rsFs,Params.WinSize,Params.HopSize,Nchannel,LowestFreq,HighestFreq,0);

% Order reverse
CentFreq = flipud(CentFreq);
LogCentFreq = log2(CentFreq);

% Log scaling of amplitude (Decibel)
if Params.LogSwitch
    Coch = 20*log10(Coch);
end

% Resample data with EqualSpacing
if Params.EqualSpacing == 1
    bin = ( log2(HighestFreq) - log2(LowestFreq) ) / Nchannel;
    LogCf = [(log2(LowestFreq)+bin/2) :bin: (log2(HighestFreq)-bin/2 )]';
    Cf = 2 .^ LogCf; 

    xx = [log2(LowestFreq):0.001:log2(HighestFreq)]';
    Idx = knnsearch(xx,LogCf); 
    nLogCf = xx(Idx);
    
    nCoch = zeros(size(Coch));
    for tt = 1:size(Coch,2)       
        yy = spline(LogCentFreq,Coch(:,tt),xx);     
        nCoch(:,tt) = yy(Idx);
    end
    LogCentFreq = nLogCf;
    CentFreq = 2 .^ LogCentFreq;
    Coch = nCoch;
end

% Average difference ofLog Central frequencies
LogScale = mean(abs(gradient(LogCentFreq)));


TimePlot = Params.HopSize*[0:size(Coch,2)] ;

if ImageSwitch
    
    CentFreq = flipud(CentFreq);
    
    FreqPos = []; TimePos = [];
    %to arrage axis values
    FreqPosData = [20 , 1000, 10000];
    for knn = 1:length(FreqPosData)
        FreqPos(knn) = knnsearch(flipud(CentFreq),FreqPosData(knn));
    end
    
    TimePosData = [0,3,6,9,12,15];
    for knn = 1:length(TimePosData)
        TimePos(knn) = knnsearch(TimePlot',TimePosData(knn));
    end
    TimePos(end) = TimePos(end)-1;
    
    figure
    imagesc(Coch);
    %colormap('parula') 
    axis xy
    caxis([-80 10])
    colorbar
    set(gca,'YTick',FreqPos)
    %set(gca,'YTickLabel',round(FreqPosData),'YDir','Normal');
    set(gca,'YTickLabel',round(FreqPosData));
    ylabel('Frequency [Hz]');
    set(gca,'XTick',TimePos)
    set(gca,'XTickLabel',round(TimePlot(TimePos)));
    xlabel('Time [s]');
    title('Gammatonegram - accurate method')
end


end

