% 创建跟踪任务视频的单独脚本。 Separate script to create a movie demonstrating the pursuit task
% Author : Hirak J Kashyap, UC Irvine
disp('Clearing workspace');
clear;

target = "sine"; % whether to run the predicitve "sine" exp. or initiation "ramp"exp.
amp = 0.7;
step = 16;       % Units of dt in a simulation step
fps = 10;
trial = 1;

if target == "sine"
    filename = strcat("pursuit_",target,"_",num2str(amp),".mat"); % save output to this file
    yMin = -amp;
    yMax = amp;

elseif target == "ramp"
    filename = strcat("pursuit_",target,"_",num2str(amp),".mat");
    yMin = -5;
    yMax = 30;
end
load(filename)

true_p = cumsum(ft);
pred_p = cumsum(zt_t(:,:,trial));
%%
% for print
linewidth = 3;
fontsize = 14;
fontweight = 'bold';

videoFileName = fullfile('output', strcat(extractBefore(filename, '.mat'),".avi") );
vidObj = VideoWriter(char(videoFileName));
vidObj.Quality = 50;
vidObj.FrameRate = fps;
open(vidObj);

figure;
plotEnd = 940;
plotStart = floor((7/15)*plotEnd);%1;

colorVec = hsv(numTrials);
for frame = plotStart:5:plotEnd
    hAxis(1) = subplot(2,1,1);
    cla
    xlim([-5 20]);
    ylim([-0.3 0.7]);
    xlabel('deg');%, 'fontsize', fontsize, 'fontweight', fontweight);
    set(gca,'YTick',[]);
    %rectangle('Position',[pred_p(1,frame)-0.5 -0.1 2 0.23],'EdgeColor','r','LineWidth',3)
    %rectangle('Position',[-4.5 0.45 1 0.12],'EdgeColor','r','LineWidth',3)
    %annotation('textbox',[0.175 0.75 0.1 0.15],'String','Eye','FitBoxToText','on', 'LineStyle', 'none');
    %if frame < (blankStart/15)*plotEnd || (blankEnd/15)*plotEnd < frame
        rectangle('Position',[true_p(1,frame)-0.2 -0.02 0.5 0.05],'FaceColor','k','EdgeColor','k','LineWidth',3)
    %end
    rectangle('Position',[0 0.48 0.5 0.05],'EdgeColor','k','FaceColor','k','LineWidth',3)
    annotation('textbox',[0.3 0.75 0.1 0.15],'String','Target','FitBoxToText','on', 'LineStyle', 'none');
    title('Position');
    
    hAxis(2) = subplot(2,1,2);
    xlim([simtime(plotStart) simtime(plotEnd)]*step);
    ylim([yMin yMax]);
    set(gca,'XTick',[simtime(plotStart):1/step:simtime(plotEnd)]*step);
    xlabel('Time (s)');%, 'fontsize', fontsize, 'fontweight', fontweight);
    ylabel('Velocity (deg/s)');%, 'fontsize', fontsize, 'fontweight', fontweight);
    title('Velocity');
    %for iBlank = 1:size(blankStart,2)
    %    rectangle('Position', [blankStart(iBlank),yMin,blankEnd(iBlank)-blankStart(iBlank),(yMax-yMin)], 'FaceColor',[0.5 .5 .5],'EdgeColor','none');
    %end
    hold on;
    
    plot(simtime(plotStart:frame)*step, ft(plotStart:frame), 'linewidth', linewidth, 'color', 'black');
    %plot(simtime(plotStart:frame)*step, zt_t(:,plotStart:frame,trial), 'linewidth', linewidth, 'color', colorVec(trial,:));
   
    pos = get( hAxis(1), 'Position' );
    pos(2) = pos(2) + 0.02; % shift up
    %pos(4) = pos(4) - 0.05 ;
    set( hAxis(1), 'Position', pos ) ;
    
    drawnow
    writeVideo(vidObj, getframe(gcf));
end
hold off;

%close(gcf)
close(vidObj);
