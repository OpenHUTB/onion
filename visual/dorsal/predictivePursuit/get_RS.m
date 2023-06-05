% 绘制视网膜滑动(Retinal Slip)
% Author : Hirak J Kashyap, UC Irvine
% 之前需要运行 main_sp_predict.m
% output: rs_fig.jpg
clear;
speed=0.7;
findOptLatency = 1;
target = "sine";
accmean = [];
accsd = [];
tLatencymean = [];
tLatencysd = [];
vel500sd = [];

filename = strcat("pursuit_",target,"_",num2str(speed),".mat");
load(filename, 'zt_t', 'nsecs', 'simtime', 'step', 'ft', 'numTrials', 'amp', 'blankStart', 'blankEnd', 'delay', 'mask', 'startStim');

rs = zeros(size(zt_t));

for iTrial = 1:size(zt_t,3)
    rs(:,:,iTrial) = zt_t(:,:,iTrial)-ft;
end

%rs = rs/size(zt_t,3);

colorVec = hsv(numTrials);
figure;
set(gca,'XTick', [simtime(1):1/step:simtime(end)]*step);
plot(simtime*step, ft, '--', 'linewidth', 1, 'color', 'black');  % 1*1526, 虚线：函数目标 function target 
ylim([-1 1]);
hold on;
% for i = 1:size(rs,3)
%     plot(simtime*step, rs(:,:,i), 'linewidth', 2, 'color', colorVec(i,:));
% end
plot(simtime*step, mean(rs,3), 'linewidth', 2, 'color', 'black');  % 1*1526*10，10次尝试做平均
xlabel('Time (s)');
ylabel('RS/Target velocity (deg/s)')

%% Save figure to local file
% rs_fig = gca;
% conf
% output_file_name = fullfile(latex_home,  'imgs', 'rs_fig');  % latex image path
% print(output_file_name, '-djpeg', '-f1', '-r600');
% print -f1 -r600 -djpeg imgs/rs_fig;
