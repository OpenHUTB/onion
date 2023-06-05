
function [STRFu, STRFuq, STRFd, STRFdq] = MakeMTFmodel(Params, LogScale, ImageSwitch)
% MTF Model based on Chi et al. (2005)
% if ImageSwitch = 1, show STRF filter figures.
if ~exist('ImageSwitch')
    ImageSwitch = 1;
end


os = Params.os;
ot = Params.ot;
ps = Params.ps;
pt = Params.pt;
FreqSpan = Params.FreqSpan;
TimeSpan = Params.TimeSpan;

OctaveFreqSpan = 2^FreqSpan;
%x is Log. Frequency (octave)
x = [-OctaveFreqSpan:0.01:OctaveFreqSpan];


xx = os*x;
hs = os * (1 - xx.^2 ) .* exp(-xx.^2 /  2); %Gamma function
hhs = imag(hilbert(hs)); %Hilbert transform


%point close to 0
CFpoint = knnsearch(x',0);
NScaledPoints = floor(max(x)/LogScale);

%down sampling to (Cochleogram) Log-scaled Frequency
sx = [];
shs = [];
shhs = [];
FreqPlot=[];
for kk = 1:(NScaledPoints * 2 + 1)
   FreqPlot(kk) = LogScale * ( kk - NScaledPoints - 1);
   ScaledPoints = knnsearch(x', LogScale * ( kk - NScaledPoints - 1 ) ); 
   sx(kk) = x(ScaledPoints);
   shs(kk) = hs(ScaledPoints);
   shhs(kk) = hhs(ScaledPoints);   
end

t = [0:0.001:TimeSpan];
tt = ot*t;

%To make the STRF onset later and avoid aliasing
tt = [zeros(1,1500) , tt];
t = [0:0.001:(TimeSpan + 1.5)];

ht = ot* tt .^ 2 .* exp( -3.5 * tt ) .* sin(2*pi.*tt);
hht = imag(hilbert(ht)); % Hilbert transform


HopSize = Params.HopSize;
NScaledPoints = floor(max(t)/HopSize);
st = [];
sht = [];
shht = [];
TimePlot=[];
for kk = 1:NScaledPoints
    TimePlot(kk) = HopSize * ( kk - 1);
    ScaledPoints = knnsearch(t', HopSize * ( kk - 1 ));
    st(kk) = t(ScaledPoints);
    sht(kk) = ht(ScaledPoints);
    shht(kk) = hht(ScaledPoints);
end

%Rotation function on complex space
th = pi/2;
RotFunc = cos(th) + i*sin(th);

hirs = shs * cos(ps) + shhs * sin(ps);
hirt = sht * cos(pt) + shht * sin(pt);
hIRS = hilbert(hirs);
hIRT = hilbert(hirt);

%row to column
hIRS = hIRS';

%upward
STRFu = real(hIRS*hIRT) ;
STRFuq = real(hIRS*hIRT*RotFunc);
%To avoid aliasing at the end period
STRFu(:, 1:(1.5/HopSize)) = [];
STRFuq(:, 1:(1.5/HopSize)) = [];

%downward
STRFd = real(hIRS*conj(hIRT));
STRFdq = real(hIRS*conj(hIRT)*RotFunc);
%To avoid aliasing at the end period
STRFd(:, 1:(1.5/HopSize)) = [];
STRFdq(:, 1:(1.5/HopSize)) = [];

TimePlot(TimePlot<1.5)=[];
TimePlot = TimePlot - 1.5;

%normalization and rounding
M = mean(reshape(STRFu, [], 1));
SD = std(reshape(STRFu, [], 1));
STRFu = decround( ( (STRFu - M ) / SD ) , 4);   

M = mean(reshape(STRFd, [], 1));
SD = std(reshape(STRFd, [], 1));
STRFd = decround( ( (STRFd - M ) / SD ) , 4); 

M = mean(reshape(STRFuq, [], 1));
SD = std(reshape(STRFuq, [], 1));
STRFuq = decround( ( (STRFuq - M ) / SD ) , 4);   

M = mean(reshape(STRFdq, [], 1));
SD = std(reshape(STRFdq, [], 1));
STRFdq = decround( ( (STRFdq - M ) / SD ) , 4);   


%if ImageSwitch is 1, display MTF images
if ImageSwitch
    
    %to arrage axis values
    for knn = 1:5
        FreqPos(knn) = knnsearch(FreqPlot',knn-3);
    end

    for knn = 1:3
        TimePos(knn) = knnsearch(1000*TimePlot',50*(knn-1));
    end

    figure
    subplot(222)
    imagesc(STRFu)
    set(gca,'XTick',TimePos)
    set(gca,'XTickLabel',round(1000*TimePlot(TimePos)));
    xlabel('Time (ms)')
    caxis([-5 5])
    %set(gca,'YTickLabel',FreqPlot(get(gca,'YTick')),'YDir','Normal');
    set(gca,'YTick',FreqPos)
    set(gca,'YTickLabel',round(FreqPlot(FreqPos)),'YDir','Normal');

    ylabel('Frequency (Octave)')
    title('STRF up')

    subplot(221)
    imagesc(STRFd)
    set(gca,'XTick',TimePos)    
    set(gca,'XTickLabel',round(1000*TimePlot(TimePos)));
    xlabel('Time (ms)')
    %set(gca,'YTickLabel',FreqPlot(get(gca,'YTick')),'YDir','Normal');
    caxis([-5 5])
    set(gca,'YTick',FreqPos)
    set(gca,'YTickLabel',round(FreqPlot(FreqPos)),'YDir','Normal');
    ylabel('Frequency (Octave)')
    title('STRF down')
    
    subplot(224)
    imagesc(STRFuq)
    set(gca,'XTick',TimePos)    
    set(gca,'XTickLabel',round(1000*TimePlot(TimePos)));
    xlabel('Time (ms)')
    %set(gca,'YTickLabel',FreqPlot(get(gca,'YTick')),'YDir','Normal');
    caxis([-5 5])
    set(gca,'YTick',FreqPos)
    set(gca,'YTickLabel',round(FreqPlot(FreqPos)),'YDir','Normal');

    ylabel('Frequency (Octave)')
    title('STRF up quadrant')

    subplot(223)
    imagesc(STRFdq)
    set(gca,'XTick',TimePos)    
    set(gca,'XTickLabel',round(1000*TimePlot(TimePos)));
    xlabel('Time (ms)')
    %set(gca,'YTickLabel',FreqPlot(get(gca,'YTick')),'YDir','Normal');
    caxis([-5 5])
    set(gca,'YTick',FreqPos)
    set(gca,'YTickLabel',round(FreqPlot(FreqPos)),'YDir','Normal');
    ylabel('Frequency (Octave)')
    title('STRF down quadrant')
    
end


function y = decround(x,n)
%round x at 10^(-n)
unit = 10^(n-1);
y = (round(x * unit)) / unit;