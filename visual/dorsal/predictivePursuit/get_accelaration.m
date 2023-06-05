% 在实验中计算平均加速度。This is a script to calculate avergae accelaration over trials
% output: accelaration_fig.jpg
% Author : Hirak J Kashyap, UC Irvine
% clear;
speeds=[-70 -65 -60 -55 -50 -45 -40 -35 -30 -25 -20 -15 -10 -5 5 10 15 20 25 30 35 40 45 50 55 60 65 70];
findOptLatency = 1;
target = "ramp";
accmean = [];
accsd = [];
tLatencymean = [];
tLatencysd = [];
vel500sd = [];
for s=speeds
    filename = fullfile('output', strcat("pursuit_",target,"_",num2str(s),".mat"));
    load(filename, 'zt_t', 'nsecs', 'simtime', 'step', 'ft', 'numTrials', 'amp', 'blankStart', 'blankEnd', 'delay', 'mask', 'startStim');

    dtStartStim = ceil((startStim/nsecs)*length(simtime));
    dtVel500 = ceil(((startStim + (0.05/step))/nsecs)*length(simtime));
    
    % determine the pursuit onset points
    acc = [];
    tLatency = [];
    vel500 = [];
    for iTrials = 1:numTrials
        if findOptLatency
            xdata = simtime(dtStartStim:dtStartStim+20)*step;
            ydata = zt_t(1,dtStartStim:dtStartStim+20,iTrials);
            fun = @(x,xdata)((x(1)<=xdata).*(x(2)*(xdata-x(1))));
            x0 = [startStim*step+0.12,s*3];
            x = lsqcurvefit(fun,x0,xdata,ydata);
            onset = x(1);
            tLatency = [tLatency,onset-(startStim*step)];            
        else
            disp direct_latency_calculation
            nz_zt = find(zt_t(1,:,iTrials));
            onset = simtime(nz_zt(1))*step;
            tLatency = [tLatency,onset-(startStim*step)];
        end
        
        % calculate average acclerartion during 80ms to 180 ms after pursuit
        % onset
        
        
        startAccPeriod = (onset+0.08)/step;
        endAccPeriod = (onset+0.18)/step;
        dtStartAccPeriod = ceil((startAccPeriod/nsecs)*length(simtime));
        dtEndAccPeriod = ceil((endAccPeriod/nsecs)*length(simtime));
        
        xdata = simtime(dtStartAccPeriod:dtEndAccPeriod)*step;
        ydata = zt_t(1,dtStartAccPeriod:dtEndAccPeriod,iTrials);
        x0 = s*3;
        v0 = ydata(1);
        fun = @(x,xdata)(v0 + x(1)*(xdata-(onset+0.08)));
        x = lsqcurvefit(fun,x0,xdata,ydata);
        acc = [acc,x(1)];
        
        % record the velocity after 500 ms from the start of the stimulus
        vel500 = [vel500,zt_t(1,dtVel500,iTrials)];        
    end
    
    accmean = [accmean,mean(acc)];
    accsd = [accsd,std(acc)];
    tLatencymean = [tLatencymean,mean(tLatency)];
    tLatencysd = [tLatencysd,std(tLatency)];
    vel500sd = [vel500sd,std(vel500)];
end
%%
% Set up experimental data from de Brouwer et al.
expSpeeds = [-45 -40 -35 -30 -25 -20 -15 -10 -5 5 10 15 20 25 30 35 40 45 50];
expMean = [-163 -125 -128 -116 -110 -100 -72 -59 -37 31 67 77 100 112 127 120 123 127 159];
expSDU = [-118 -88 -97 -85 -78 -71 -46 -26 -21 68 105 110 146 157 181 177 184 180 216];

expSD = (expSDU - expMean) * 2;

figure;
errorbar(expSpeeds, expMean, expSD, 'r.','MarkerSize', 20, 'linewidth', 2);
hold on;
%plot the ramp velocity vs acceleration (mean and SD) plot 
errorbar(speeds, accmean, accsd, 'b.','MarkerSize', 20, 'linewidth', 2);
%plot(speeds, accmean, '.b', 'MarkerSize',20);
legend({'Experimental data', 'Model'}, 'Location','southeast','FontSize',12);
xlabel('Target velocity (deg/s)','FontSize',12);
ylabel('Acceleration (deg/s^2)','FontSize',13);

%% Save figure to local file
% accelaration_fig = gca;
% output_file_name = 'D:/doc/thesis/2_BTN/latex/imgs/accelaration_fig';  % latex image path
% print(output_file_name, '-djpeg', '-f1', '-r600');

%%
%     linewidth = 1;
%     fontsize = 14;
%     fontweight = 'bold';
% 
%     colorVec = hsv(numTrials);
%     yMin = -amp*2;
%     yMax = amp*2;
%     figure;
%     ylim([yMin yMax]);
%     set(gca,'XTick',[simtime(1):0.02:simtime(end)]*step);
%     for iBlank = 1:size(blankStart,2)
%         rectangle('Position', [blankStart(iBlank),yMin,blankEnd(iBlank)-blankStart(iBlank),(yMax-yMin)], 'FaceColor',[0.5 .5 .5],'EdgeColor','none');
%     end
%     hold on;
%     plot(simtime*step, ft, 'linewidth', linewidth, 'color', 'black');
%     for trial=1:numTrials
%         plot(simtime*step, zt_t(:,:,trial), 'linewidth', linewidth, 'color', colorVec(trial,:));
%     end
% 
%     %title('', 'fontsize', fontsize, 'fontweight', fontweight);
%     xlabel('Time (s)', 'fontsize', fontsize, 'fontweight', fontweight);
%     ylabel('Pursuit velocity (rad/s)', 'fontsize', fontsize, 'fontweight', fontweight);
%     hold off;
