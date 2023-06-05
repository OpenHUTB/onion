% clear;
% wrong: Undefined function or variable 'wr_sample100'.
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

colorVec = hsv(100);
figure;
set(gca,'XTick',[simtime(1):1/step:simtime(end)]*step);
ylim([-.25 .25]);
hold on;
for i = 1:100
    plot(simtime*step, wr_sample100(i,:,1), 'linewidth', 2, 'color', colorVec(i,:));
end
xlabel('Time (s)');
ylabel('RS/Target velocity (deg/s)')
