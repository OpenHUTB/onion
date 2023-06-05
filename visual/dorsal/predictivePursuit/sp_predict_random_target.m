% Script to run the predictive smooth pursuit of random target
% Author : Hirak J Kashyap, UC Irvine
% Based on FORCE learning code by David Sussillo (Neuron 2009)
% 输出：velocity_fig.jpg
disp('Clearing workspace');
clear;

rng(0);          % For reproducibility -- any seed is fine
target = "sine"; % whether to run the predicitve "sine" exp. or initiation "ramp"exp.
numTrials = 10;  % #trials of the experiment
feedback = 1;    % Output feedback to the RNN 0/1
N = 500;         % Number of neurons in RNN
p = 1.0;         % 1/width of initial weight distribution
dt = 0.001;
dtt = dt * 50;
step = 16;       % Units of dt in a simulation step
g = 1.5;		 % g greater than 1 leads to chaotic networks.
wf = 2.0*(rand(N,1)-0.5); % Feedback weights
delay = 80/step; % Length of biophysical input delay 
learn_every = delay;

if target == "sine"
    nsecs = 22; % Experiment time in seconds
    blankStart = [12.5 16 19]; % start time(s) of blank period(s) separated by comma(s)
    blankEnd = [13 17 20.5];   % end time(s) of blank period(s) separated by comma(s)
    startStim = 0;     % Simulation start time
    nsecs = nsecs/step;
    gain = 2;          % leaky-integrator gain
    alpha = 100;       % P matrix initialized to 1/alpha
    freq = 0.5*3;      % Frequency of the sinusoidal target
    amp = 0.7;         % Amplitude of the sinusoidal target
    
    simtime = 0:dt:nsecs-dt;
    simtime_len = length(simtime);
    %ft = (amp/1.0)*sin(32*pi*freq*simtime);  % Target function
    ft = (amp/1.0)*sin(1.0*pi*freq*simtime) + ...
     (amp/2.0)*sin(2.0*pi*freq*simtime) + ...
     (amp/6.0)*sin(3.0*pi*freq*simtime) + ...
     (amp/3.0)*sin(4.0*pi*freq*simtime);
    ft = ft/1.5;
    filename = strcat("pursuit_",target,"_",num2str(amp),".mat"); % save output to this file
    yMin = -1;%-abs(amp)*2;
    yMax = 1;%abs(amp)*2;

elseif target == "ramp"
    nsecs = 4;
    startStim = 0.4;
    blankStart = [];
    blankEnd = [];
    
    nsecs = nsecs/step;
    startStim = startStim/step;
    gain = 0.5;
    alpha = 1.25;
    amp =19;
    
    simtime = 0:dt:nsecs-dt;
    simtime_len = length(simtime);
    dtStartStim = floor((startStim/nsecs)*length(simtime));
    ft = [zeros(1,dtStartStim),ones(1,simtime_len-dtStartStim)].*amp;
    filename = strcat("pursuit_",target,"_",num2str(amp),".mat");
    yMin = -5;
    yMax = 30;
end

tm = 8; % Time constant of the leaky integrator

% Create mask for the blank period(s)
mask = zeros(1, length(simtime)); 
for iBlank = 1:size(blankStart,2)
    dtStart = floor(((blankStart(iBlank)/step)/nsecs)*length(simtime));
    dtEnd = floor(((blankEnd(iBlank)/step)/nsecs)*length(simtime));
    mask(dtStart:dtEnd) = 1;
end

% for print
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
    % initialize all the dynamic states
    li = 0; % leaky integrator state
    eig_written = 0;
    
    M = randn(N,N)*g*scale; % RNN weight matrix
    wo = zeros(nRec2Out,1); % readout weight matrix
    dw = zeros(nRec2Out,1); % weight change

    wo_len = zeros(1,simtime_len);    
    rPr_len = zeros(1,simtime_len);
    zt = zeros(1,simtime_len); % output predicition signal

    x0 = 0.5*randn(N,1); % neuron states
    z0 = 0.5*randn(1,1); % prediction start

    x = x0; 
    r = tanh(x);         % Non-linearity of neurons
    xp = x0;
    z = z0; 
    e = 0;                         % error initialization
    ti = 0;                        % count of time steps
    P = (1.0/alpha)*eye(nRec2Out); % Learning rate matrix of FORCE learning
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

        if mod(ti-1, learn_every) == 0 && ~mask(ti)
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
    end
    error_avg_train = sum(abs(zt-ft))/simtime_len;
    wo_len_t(:,:,trial) = wo_len;
    zt_t(:,:,trial) = zt;
    error_avg_train_t(:,trial) = error_avg_train;
end

%%
colorVec = hsv(numTrials);
figure;
ylim([yMin yMax]);
xlim([0 22]);
set(gca,'XTick',[simtime(1):1/step:simtime(end)]*step);
for iBlank = 1:size(blankStart,2)
    rectangle('Position', [blankStart(iBlank),yMin,blankEnd(iBlank)-blankStart(iBlank),(yMax-yMin)], 'FaceColor',[0.5 .5 .5],'EdgeColor','none');
end
hold on;
plot(simtime*step, ft, '--', 'linewidth', linewidth, 'color', 'black');

for trial=1:numTrials
    plot(simtime*step, zt_t(:,:,trial), 'linewidth', linewidth, 'color', colorVec(trial,:));
end
    
%title('', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('Time (s)');%, 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('Target/Eye velocity (deg/s)');%, 'fontsize', fontsize, 'fontweight', fontweight);
hold off;
save(filename, 'zt_t', 'nsecs', 'simtime', 'step', 'ft', 'numTrials', 'amp', 'blankStart', 'blankEnd', 'delay', 'mask', 'startStim');

%% Save figure to local file
% velocity_fig = gca;
% conf
% output_file_name =fullfile(latex_home, 'imgs', 'velocity_fig');  % latex image path
% print(output_file_name, '-djpeg', '-f1', '-r600');
