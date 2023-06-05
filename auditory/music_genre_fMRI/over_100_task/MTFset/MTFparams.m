function Params = MTFparams(varargin)
%MTF parameters
% N (feature dimension) equals to size(osParams)*size(otParams)*size(NFreqFeature)
%
%Nchannel...Number of frequency filters to extract gammatone spectrogram 
%WinSize...Time window size
%HopSize...Each time window hop(短距离跳跃) size (hop windows: 滑动窗口)
%          The window size was set to 25 ms, with the hop size set to 10 ms.
%LowestFreq... Lowest frequency of spectrogram
%HighestFreq... Highest frequency of spectrogram
%LogSwitch... if 1, spectrogram returns log value.
%EqualSpacing... if 1, take equal spacing in the logarithmic frequency space
%osParams...Frequent rate range (cycles/Octave) 
%otParams...Temporal rate range (Hz)
%os...Frequency (cycles/Octave)
%ot...Temporal rate (Hz)
%ps...Symmetry parameter of Gabor function
%pt...Symmetry paramter of Gamma function
%FreqSpan...Frequency range of individual STRF filter
%TimeSpan...Time range of individual STRF filter
%NFreqFeature...Number of central frequency took
%sqrtSwitch...if 1, take square root of Modulation evergy
%LogSwitch2...if 1, take log of Modulation energy (after sqrt)
%FreqMethod...'Mean': average around the central frequency. 'Central': only
%take output from the cental frequency
%cNorm...if 1, normalize wave amplitude before making cochleogram
%udSwitch...if 1, output both upper and down modulatin, if 0, average them

if nargin == 0
   Params = [];
   MTFCode = 1;
end

if nargin == 1
   MTFCode = varargin{1};
end

if nargin == 2
    MTFCode = varargin{1};
    Params = varargin{2};    
end

Params.Nchannel = 120;
Params.WinSize = 0.025;
Params.HopSize = 0.010;
Params.LowestFreq = 20; 
Params.HighestFreq = 10000;
Params.LogSwitch = 1; 
Params.EqualSpacing = 1;
Params.osParams = [0.5 1 2 4 8];
Params.otParams = [2 4 8 16 32]; 
Params.os = 1; 
Params.ot = 16;
Params.ps = 0; 
Params.pt = 0; 
Params.FreqSpan = 1.25;
Params.TimeSpan = 0.125;
Params.NFreqFeature = 20; 
Params.FreqMethod = 'mean';
Params.sqrtSwitch = 1;
Params.LogSwitch2 = 1;
Params.cNorm = 1;
Params.udSwitch = 1;

switch MTFCode 
    case {1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 22,24,26, 28}
        Params.FreqMethod = 'mean';
        switch MTFCode 
            case 1
               Params.sqrtSwitch = 1;
               Params.LogSwitch2 = 0;
            case 2
               Params.sqrtSwitch = 0;                        
               Params.LogSwitch2 = 1;
            case 3
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
            case 8
               Params.sqrtSwitch = 0;
               Params.LogSwitch2 = 0;                          
            case 9
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;                       
               Params.LogSwitch = 0; 
            case 10
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;  
               Params.WinSize = 0.005;
               Params.HopSize = 0.002;
            case 11
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
               Params.NFreqFeature = 100; %total N features = 5000
            case 12
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
               Params.FreqSpan = 3;
               Params.TimeSpan = 0.5;                       
               Params.osParams = [0.25 0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
            case 13
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
               Params.FreqSpan = 3;
               Params.TimeSpan = 0.5;                       
               Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];                        
            case 14
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
               Params.FreqSpan = 3;
               Params.TimeSpan = 0.5;
               Params.osParams = [0.25 0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];
            case 15
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
               Params.osParams = [0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];
            case 17 
               Params.cNorm = 0; 
            case 19
               Params.udSwitch = 0;
            case 20
               Params.Nchannel = 128;
            case 22
               Params.Nchannel = 128;
               Params.NFreqFeature = 10;             
               Params.udSwitch = 0;
               Params.osParams = [0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [2.8 4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0];
            case 24
               Params.Nchannel = 128;
               Params.NFreqFeature = 20;             
               Params.udSwitch = 0;
               % Modulation-selective filters were tuned to 10 spectral modulation scales 
               % [Ω = (0.35, 0.50, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0,5.66, 8.0) cyc/oct]
               % and 10 temporal modulation rates 
               % [ω = (2.8, 4.0,5.7, 8.0, 11.3, 16.0, 22.6, 32.0, 45.3, 64.0) Hz].
               Params.osParams = [0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [2.8 4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0];
            case 26
               Params.Nchannel = 128;                
               Params.FreqSpan = 3;
               Params.TimeSpan = 0.5;
               Params.udSwitch = 0;               
               Params.osParams = [0.25 0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];
            case 28
               Params.Nchannel = 128;                  
               Params.udSwitch = 0;               
               Params.osParams = [0.25 0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];                               
        end        
    case {4, 5, 6, 21, 23, 25, 27}
        Params.FreqMethod = 'central';
        switch MTFCode 
            case 4
               Params.sqrtSwitch = 1;
               Params.LogSwitch2 = 0;
            case 5
               Params.sqrtSwitch = 0;                        
               Params.LogSwitch2 = 1;
            case 6
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
            case 21
               Params.Nchannel = 128;
            case 23
               Params.Nchannel = 128;
               Params.NFreqFeature = 10;             
               Params.udSwitch = 0;
               Params.osParams = [0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [2.8 4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0];
            case 25
               Params.Nchannel = 128;
               Params.NFreqFeature = 20;             
               Params.udSwitch = 0;
               Params.osParams = [0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [2.8 4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0];
            case 27
               Params.Nchannel = 128;                
               Params.sqrtSwitch = 1;                        
               Params.LogSwitch2 = 1;
               Params.FreqSpan = 3;
               Params.TimeSpan = 0.5;
               Params.udSwitch = 0;               
               Params.osParams = [0.25 0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
               Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];                               
                
        end

    case {7, 16, 18}
        Params.FreqMethod = 'None';
        switch MTFCode
            case 7
                Params.sqrtSwitch = 1;                        
                Params.LogSwitch2 = 1;
            case 16
                Params.sqrtSwitch = 1;                        
                Params.LogSwitch2 = 1;
                Params.FreqSpan = 3;
                Params.TimeSpan = 0.5;
                Params.osParams = [0.25 0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
                Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];
            case 18
                Params.sqrtSwitch = 1;                        
                Params.LogSwitch2 = 1;
                Params.LogSwitch = 0;
                Params.FreqSpan = 3;
                Params.TimeSpan = 0.5;
                Params.osParams = [0.25 0.35 0.5 0.71 1.00 1.41 2.00 2.83 4.00 5.66 8.00];
                Params.otParams = [4.0 5.7 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0];      
        end

end


