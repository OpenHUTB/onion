function Features = ExtractMTFfeatures(sCoch, LogScale, FileLength, TR, MTFCode)

Params = MTFparams(MTFCode);

NFreqFeature = Params.NFreqFeature;
FreqMethod = Params.FreqMethod;
HopSize = Params.HopSize;
osParams = Params.osParams; 
otParams = Params.otParams;
sqrtSwitch = Params.sqrtSwitch;
LogSwitch2 = Params.LogSwitch2;
bin = TR/HopSize;

Features = [];
for pos = 1:length(osParams) 
    for pot = 1:length(otParams)

        Params.os = osParams(pos);
        Params.ot = otParams(pot);

        % Make STRF filters (up/down, quadrature pair)
        [STRFu, STRFuq, STRFd, STRFdq] = MakeMTFmodel(Params, LogScale, 0);

        % Zero Padding of cochleogram
        PadSize = floor(size(STRFu)/2) + 1;
        pdsCoch = padarray(sCoch,PadSize);

        % Convolution with STRF filters
        uConv = conv2(pdsCoch,STRFu);
        uqConv = conv2(pdsCoch,STRFuq);
        dConv = conv2(pdsCoch,STRFd);
        dqConv = conv2(pdsCoch,STRFdq);

        % Cropping, if STRFu matrix size (either row/column) is even, crop 1
        % more row/column from the end        
        uConv = EvenOddCrop(uConv,STRFu);
        uqConv = EvenOddCrop(uqConv,STRFuq);
        dConv = EvenOddCrop(dqConv,STRFd);
        dqConv = EvenOddCrop(dqConv,STRFdq);        

        % Calculate energy
        uEnergy = uConv .* uConv + uqConv .* uqConv;
        dEnergy = dConv .* dConv + dqConv .* dqConv;

        % Switch whether taking root of energy 
        if sqrtSwitch
            uEnergy = sqrt(uEnergy);
            dEnergy = sqrt(dEnergy);
        end

        % Switch whether taking log of energy
        if LogSwitch2
           uEnergy = log(uEnergy);
           dEnergy = log(dEnergy);
        end
        
        % Average for time (TR)
        % Duplicate the first and last colomn to match the column size
        uEnergy = [uEnergy(:,1), uEnergy, uEnergy(:,end)];
        dEnergy = [dEnergy(:,1), dEnergy, dEnergy(:,end)];        
        muEnergy = [];
        mdEnergy = [];
        for TT = 1:(FileLength/TR)            
            muEnergy = [muEnergy, mean(uEnergy(:,(1+bin*(TT-1)):TT*bin),2)];
            mdEnergy = [mdEnergy, mean(dEnergy(:,(1+bin*(TT-1)):TT*bin),2)];
        end


        FreqAveSize = floor( size(muEnergy,1) / NFreqFeature) ;  
        % Extraction around central frequency
        switch FreqMethod 
            case 'mean'      
                for Freq = 1:NFreqFeature
                    FreqStart = round( ( Freq - 1 ) * size(muEnergy,1) / NFreqFeature );
                    muEnergyPart = muEnergy( ( FreqStart + 1):( FreqStart + FreqAveSize ), :);
                    mdEnergyPart = mdEnergy( ( FreqStart + 1):( FreqStart + FreqAveSize ), :);
                    if Params.udSwitch == 1
                        Features =[Features; mean(muEnergyPart,1); mean(mdEnergyPart,1)];
                    else
                        udFeat = [mean(muEnergyPart,1);  mean(mdEnergyPart,1)];
                        Features =[Features; mean(udFeat,1)];                       
                    end
                end       
            case 'central'
                for Freq = 1:NFreqFeature
                    FreqStart = round( ( Freq - 1 ) * size(muEnergy,1) / NFreqFeature );
                    muEnergyPart = muEnergy( FreqStart + ceil(FreqAveSize/2), :);
                    mdEnergyPart = mdEnergy( FreqStart + ceil(FreqAveSize/2), :);
                    if Params.udSwitch == 1                    
                        Features =[Features; muEnergyPart; mdEnergyPart];
                    else
                        udFeat = [muEnergyPart;  mdEnergyPart];
                        Features =[Features; mean(udFeat,1)];                                  
                    end
                end
            case 'None'
                mudEnergy = [muEnergy; mdEnergy];
                if Params.udSwitch == 1                  
                    Features = [Features; mudEnergy];
                else
                    udFeat = [muEnergy; mdEnergy];
                    Features =[Features; mean(udFeat,1)];                        
                end
        end

    end
end




function uConv = EvenOddCrop(uConv, STRFu)
        % Cropping, if STRFu matrix size (either row/column) is even, crop 1
        % more row/column from the end
        uConv(1:size(STRFu,1),:) = [];
        uConv((size(uConv,1)-size(STRFu,1)+1 - rem(size(STRFu,1)+1,2) ):end,:)=[];
        uConv(:,1:size(STRFu,2)) = [];
        uConv(:,(size(uConv,2)-size(STRFu,2)+1 - rem(size(STRFu,2)+1,2) ):end)=[];
        
            
