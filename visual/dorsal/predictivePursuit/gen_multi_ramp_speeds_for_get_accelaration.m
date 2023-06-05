% 为get_accelaration准备数据，数据输出到output文件夹中。
% 作者：王海东
% 基于 : Hirak J Kashyap, UC Irvine
% Based on FORCE learning code by David Sussillo (Neuron 2009)
disp('Clearing workspace');
clear;

speeds=[-70 -65 -60 -55 -50 -45 -40 -35 -30 -25 -20 -15 -10 -5 5 10 15 20 25 30 35 40 45 50 55 60 65 70];

rng(0);          % 为了实验复现（任何种子都可以）For reproducibility -- any seed is fine
target = "ramp";
% target = "sine"; % 是否运行预测性正弦指数，或者初始ramp指数。whether to run the predicitve "sine" exp. or initiation "ramp"exp.
numTrials = 10;  % 实验的尝试次数。trials of the experiment
feedback = 1;    % RNN的输出反馈（考虑反馈）。 Output feedback to the RNN 0/1
N = 500;         % RNN中神经元的个数。 Number of neurons in RNN
p = 1.0;         % 初始权重分布宽度的倒数。 1/width of initial weight distribution
dt = 0.001;
dtt = dt * 100;
step = 16;                % 一个仿真步中dt的单位。 Units of dt in a simulation step
% 神经系统具有分叉、混沌和奇怪吸引子动力学行为。
% 混沌神经网络模型是在Hopfield神经网络中引入一个具有混沌特性的负反馈项。
% 循环网络的权重初始化：均值为0，标注差为1.5的高斯分布，导致一种自发地混沌行为。
g = 1.5;                  % g大于1表示是一个混沌网络。 g greater than 1 leads to chaotic networks.
wf = 2.0*(rand(N,1)-0.5); % 反馈权重(500*1)。 Feedback weights
delay = 80/step;          % 生物物理学 输入延迟的长度(5)。 Length of biophysical input delay 
learn_every = delay;

if target == "sine"
    nsecs = 25;        % 实验时间（秒） Experiment time in seconds
    blankStart = [15]; % 空白周期的开始时间（以逗号分隔）。 start time(s) of blank period(s) separated by comma(s)
    blankEnd = [20];   % end time(s) of blank period(s) separated by comma(s)
    startStim = 0;     % 仿真开始时间。Simulation start time
    nsecs = nsecs/step;
    gain = 1;          % 漏积分器（低通滤波） 增益。 leaky-integrator gain
    alpha = 100;       % P矩阵初始化为alpha的倒数。 P matrix initialized to 1/alpha
    freq = 0.5;        % 正弦曲线目标的频率。 Frequency of the sinusoidal target
    amp = 0.7;         % 正弦曲线目标的振幅。 Amplitude of the sinusoidal target
    
    simtime = 0:dt:nsecs-dt;
    simtime_len = length(simtime);
    ft = (amp/1.0)*sin(32*pi*freq*simtime);  % 目标函数。 Target function
    ft = ft/1.5;
    filename = strcat("pursuit_",target,"_",num2str(amp),".mat"); % save output to this file
    yMin = -abs(amp)*2;
    yMax = abs(amp)*2;

elseif target == "ramp"
    for amp = speeds
        nsecs = 4;
        startStim = 0.4;
        blankStart = [];
        blankEnd = [];
        
        nsecs = nsecs/step;
        startStim = startStim/step;
        gain = 0.5;
        alpha = 1.25;
%         amp =19;
        
        simtime = 0:dt:nsecs-dt;
        simtime_len = length(simtime);
        dtStartStim = floor((startStim/nsecs)*length(simtime));
        ft = [zeros(1,dtStartStim),ones(1,simtime_len-dtStartStim)].*amp;
        filename = fullfile('output', strcat("pursuit_",target,"_",num2str(amp),".mat"));
        yMin = -5;
        yMax = 30;
        
        
        %% 
        % 漏积方程是一种特殊的微分方程，用于描述对输入进行积分但随着时间的推移逐渐 漏出少量输入 的组件或系统。
        % 这相当于一个一阶低通滤波器，其截止频率远低于感兴趣的频率。
        % 它通常出现在水力学、电子学和神经科学中。
        % ref: https://en.wikipedia.org/wiki/Leaky_integrator
        tm = 8; % 低通滤波积分器的时间常数。 Time constant of the leaky integrator
        
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
        output_wt_t = zeros(nRec2Out, 1, numTrials);  % 读出权重使用0进行初始化
        
        for trial = 1:numTrials
            trial
            % 初始化所有的动力学状态。 initialize all the dynamic states
            li = 0; % 漏积分器 状态。 leaky integrator state
            eig_written = 0;
            
            M = randn(N,N)*g*scale; % RNN权重矩阵(500*500)。 RNN weight matrix
            wo = zeros(nRec2Out,1); % 读出权重矩阵(500*1)。 readout weight matrix
            dw = zeros(nRec2Out,1); % 权重改变（500*1)。 weight change
            
            wo_len = zeros(1,simtime_len);
            rPr_len = zeros(1,simtime_len);
            zt = zeros(1,simtime_len);      % output predicition signal
            
            x0 = 0.5*randn(N,1); % 神经元状态(500*1)。 neuron states
            z0 = 0.5*randn(1,1); % 预测值的开始位置（随机）。 prediction start
            
            x = x0;
            r = tanh(x);         % 神经元的非线性（双曲正切(-1,1)：双曲正弦与双曲余弦的比值）。 Non-linearity of neurons
            xp = x0;
            z = z0;
            e = 0;                         % 误差初始化。 error initialization
            ti = 0;                        % 时间步计数。count of time steps
            P = (1.0/alpha)*eye(nRec2Out); % FORCE学习（用于RNN的训练）的学习率矩阵(500*500)。 Learning rate matrix of FORCE learning
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
                li = li + 1/tm * (- li + gain*z);  % 解微分方程（漏积分器、低通滤波）
                
                if mod(ti-1, learn_every) == 0 && ~mask(ti)
                    % 更新相关矩阵的逆阵。 update inverse correlation matrix
                    k = P*r;
                    rPr = r'*k;
                    c = 1.0/(1.0 + rPr);
                    P = P - k*(k'*c);
                    
                    % 更新线性读出的误差。 update the error for the linear readout
                    if ti > delay
                        e = zt(ti-delay) - ft(ti-delay);
                    else
                        e = 0;
                    end
                    
                    % 更新输出权重。 update the output weights
                    dw = -e*k*c;
                    wo = wo + dw;
                    
                    % 用输出的误差更新内部权重矩阵。 update the internal weight matrix using the output's error
                    M = M + repmat(dw', N, 1);
                end
                
                % 存下系统的输出。 Store the output of the system.
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


        
    end
end


