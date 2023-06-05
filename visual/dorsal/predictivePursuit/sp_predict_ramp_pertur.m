% Script to run the predictive smooth pursuit during perturbations and
% phase shift
% Author : Hirak J Kashyap, UC Irvine
% Based on FORCE learning code by David Sussillo (Neuron 2009)
% 输出: perturbation_fig.jpg
disp('Clearing workspace');
clear;

rng(0);
target = "sine";
% target = "ramp";  % 对应跟踪初始化？ Pursuit initiation?
numTrials = 20;
feedback = 1;
N = 500;
p = 1.0;
dt = 0.001;
dtt = dt * 100;
step = 16;
g = 1.5;				% g greater than 1 leads to chaotic networks.
wf = 2.0*(rand(N,1)-0.5);
delay = 80/step;
learn_every = delay;

if target == "sine"
    nsecs = 50;
    perturStart = [14.5];
    perturEnd = [15.5]; % van den berg used half cycle perturbation - 1sec
    perturLength = perturEnd - perturStart; %add perturLength amount of ramp and shift ft to right by perturLength
    startStim = 0;
    nsecs = nsecs/step;
    
    perturStart = perturStart/step;
    perturLength = perturLength/step;
    
    gain = 5;%1;%0.75;%0.5;
    alpha = 275;%100;%80;%80;%20;%70;%45;%0.25;%1.0e-0;
    freq = 0.5;
    
    amp = 0.7;
    
    simtime = 0:dt:nsecs-2*dt;
    simtimePeriodic = 0:dt:nsecs-perturLength-dt;
    perturtime = 0:dt:perturLength-dt;
    simtime_len = length(simtimePeriodic)+length(perturtime);
    
    ftt = (amp/1.0)*sin(32*pi*freq*simtimePeriodic);
    ftt = ftt/1.5;
    
    fttbefore = ftt(1:floor(perturStart/dt));
    fttafter = ftt(ceil(perturStart/dt):end);
    
    ft = [fttbefore,ones(1,length(perturtime))*fttbefore(1,end),fttafter];
    ft = ft(1:size(simtime,2));
    
    filename = strcat("pursuit_",target,"_",num2str(amp),".mat");
    yMin = -abs(amp)*2;
    yMax = abs(amp)*2;

elseif target == "ramp"
    nsecs = 4;
    startStim = 0.4;
    perturStart = [];
    perturEnd = [];
    
    nsecs = nsecs/step;
    startStim = startStim/step;
    gain = 0.5;% this is fixed
    alpha = 1.25;%
    
    amp =20;
    simtime = 0:dt:nsecs-dt;
    simtime_len = length(simtime);
    dtStartStim = floor((startStim/nsecs)*length(simtime));
    ft = [zeros(1,dtStartStim),ones(1,simtime_len-dtStartStim)].*amp;
%     ft = [ones(1,simtime_len)].*amp;
    filename = strcat("pursuit_",target,"_",num2str(amp),".mat");
    yMin = -5;
    yMax = 30;
end

tm = 8; 

linewidth = 1;
fontsize = 14;
fontweight = 'bold';

scale = 1.0/sqrt(p*N);
nRec2Out = N;

wo_len_t = zeros(1,simtime_len, numTrials);
zt_t = zeros(1,simtime_len, numTrials);
wr_sample100 = zeros(100,simtime_len, numTrials);
error_avg_train_t = zeros(1, numTrials);

internal_wt_t = zeros(N, N, numTrials);
output_wt_t = zeros(nRec2Out, 1, numTrials);

for trial = 1:numTrials
    trial
    li = 0;
    eig_written = 0;
    
    M = randn(N,N)*g*scale;
    wo = zeros(nRec2Out,1);
    dw = zeros(nRec2Out,1);

    wo_len = zeros(1,simtime_len);    
    rPr_len = zeros(1,simtime_len);
    zt = zeros(1,simtime_len);

    x0 = 0.5*randn(N,1);
    z0 = 0.5*randn(1,1);

    x = x0; 
    r = tanh(x);
    xp = x0;
    z = z0; 
    e = 0;
    ti = 0;
    P = (1.0/alpha)*eye(nRec2Out);
    for t = simtime
        ti = ti+1;	

        % sim, so x(t) and r(t) are created.

        if(feedback == 1)
            x = (1.0-dtt)*x + M*(r*dtt)+ wf*(z*dtt);
        else
            x = (1.0-dtt)*x + M*(r*dtt);
        end
        r = tanh(x);
        z = wo'*r;
        li = li + 1/tm * (- li + gain*z);

        if mod(ti-1, learn_every) == 0
            % update inverse correlation matrix
            k = P*r;
            rPr = r'*k;
            c = 1.0/(1.0 + rPr);
            P = P - k*(k'*c);

            % update the error for the linear readout
            if ti > delay
                e = zt(ti-delay) - ft(ti-delay);
            else
                e = 0;
            end

            % update the output weights
            dw = -e*k*c;
            wo = wo + dw;

            % update the internal weight matrix using the output's error
            M = M + repmat(dw', N, 1);
        end

        % Store the output of the system.
        zt(ti) = li;
        wo_len(ti) = sqrt(wo'*wo);
%         从500*500的权重矩阵中采样出100*1的向量
%         wr_sample100(:,ti,trial) = M(wt_sample_xy);
    end
    error_avg_train = sum(abs(zt-ft))/simtime_len;
    wo_len_t(:,:,trial) = wo_len;
    zt_t(:,:,trial) = zt;
    error_avg_train_t(:,trial) = error_avg_train;
end

%%
figure;
ylim([yMin yMax]);
set(gca,'XTick',[simtime(1):1/step:simtime(end)]*step);
% for iBlank = 1:size(perturStart,2)
%     rectangle('Position', [perturStart(iBlank),yMin,perturEnd(iBlank)-perturStart(iBlank),(yMax-yMin)], 'FaceColor',[0.5 .5 .5],'EdgeColor','none');
% end
hold on;
plotStart = 625;
plotEnd = 1250;
plot(simtime(plotStart:plotEnd)*step, ft(plotStart:plotEnd), '--', 'linewidth', linewidth, 'color', 'black');
xlim([simtime(plotStart) simtime(plotEnd)]*step);
ylim([-amp amp]);
%[SE I] = sort(error_avg_train_t);
Samples = [4,6,8,9,5];%[4,6,8,9,12,5];
colorVec = hsv(max(Samples));
for trial=Samples
    plot(simtime(plotStart:plotEnd)*step, zt_t(:,plotStart:plotEnd,trial), 'linewidth', linewidth, 'color', colorVec(trial,:));
end

y=[-amp,amp];
x = [perturStart*step, perturStart*step];
plot(x,y,':k', 'LineWidth', 2);

y=[-amp,amp];
x = [perturStart*step + 0.58, perturStart*step + 0.58];
plot(x,y,':k', 'LineWidth', 2);

y=[-0.6, -0.6];
x = [perturStart*step, perturStart*step+0.58];
annotation('doublearrow',(x-simtime(plotStart)*step)/((simtime(plotEnd)-simtime(plotStart))*step),(y+amp)/(2*amp));

if target=="ramp"
    if amp>0
        y=[0,amp];
    else
        y=[-amp,0];
    end
    x=[startStim*step,startStim*step];
    %plot(x,y,'--b');
    x=x+0.2;
    %plot(x,y,'--b');
    x=x+0.2;
    %plot(x,y,'--b');
    x=x+0.2;
    %plot(x,y,'--b');
else
    y = [-amp, amp];
    x = [1, 1];
    %plot(x,y,'--b');
    x = x + 1;
    %plot(x,y,'--b');
    x = x + 1;
    %plot(x,y,'--b');
    x = x + 1;
    %plot(x,y,'--b');
end
    
%title('', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('Time (s)');%, 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('Target/Eye velocity (deg/s)');%, 'fontsize', fontsize, 'fontweight', fontweight);
hold off;
save(filename, 'zt_t', 'nsecs', 'simtime', 'step', 'ft', 'numTrials', 'amp', 'perturStart', 'perturEnd', 'delay', 'startStim');

%% Save figure to local file
% perturbation_fig = gca;
% conf
% output_file_name =fullfile(latex_home, 'imgs', 'perturbation_fig');  % latex image path
% print(output_file_name, '-djpeg', '-f1', '-r600');