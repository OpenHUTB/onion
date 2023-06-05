%% 处理猴子的数据
clear;
close all

load('monkeydata_training.mat'); %trail: 100x8 struct
trial_struct=struct2cell(trial);

a = 1; %trial number out of 100
b = 1; %1 of the 8 directions
trial_num = cell2mat(trial_struct(1,a,b));
neural_spikes = cell2mat(trial_struct(2,a,b));
hand_pos = cell2mat(trial_struct(3,a,b));

all_neural_spikes = trial_struct(2,:,:);

figure
imagesc(neural_spikes)
title('Spike plot for trial 1')
xlabel('Time')
ylabel('Neuron Number')

% raster plot
M = [];
neuron_id = 42;
LengthMax = 1000;

%%%% For a specific neuron out of the 98 neurons
figure
for b = 1:8
    for a = 1:100
        neural_spikes = cell2mat(trial_struct(2,a,b));
        neuron = neural_spikes(neuron_id,:)*b;%%%% times b so we can use colormap
        Size = size(neuron);
        neuron = [neuron zeros(1,1000-Size(2))];
        M = vertcat(M, neuron);
    end
end

imagesc(M)
str = sprintf('Raster plot for Neuron Unit %d across all trials', neuron_id); %%% There are 100 x 8 = 800 trials in total
colormap(hot)
title(str)
xlabel('time')
ylabel('trialNum')


%%%% Plot the Peri-Stimulus Time Histograms(PSTHs)
M = [];
for b = 1:8
    for a = 1:100
        neural_spikes = cell2mat(trial_struct(2,a,b));
        neuron = neural_spikes(neuron_id,:);%%%% times b so we can use colormap
        Size = size(neuron);
        neuron = [neuron zeros(1,1000-Size(2))];
        M = vertcat(M, neuron);
    end
end

figure;
N=[];
for i=1:1000
    N(i)=sum(M(:,i));
end
bar(N);



%%%% Plot hand positions for different trials
figure
for b = 1:8
    for a = 1:100
        hand_pos = cell2mat(trial_struct(3,a,b));
        X_pos = hand_pos(1,:);
        Y_pos = hand_pos(2,:);
        plot(X_pos, Y_pos);
        hold on
    end
end
hold off


%%%%% Tuning Curve: firing rate(averaged across time and trials vs.
%%%%% direction)
M = [];
for b = 1:8
    for a = 1:100
        neural_spikes = cell2mat(trial_struct(2,a,b));
        neuron = neural_spikes(neuron_id,:);%%%% times b so we can use colormap
        Size = size(neuron);
        neuron = [neuron zeros(1,1000-Size(2))];
        M = vertcat(M, neuron);
    end
end

N=[];
for i = 1:8
    N(i)=sum(sum(M((100*i-99):100*i,:)));
end
figure;
bar(N);
str = sprintf('Tuning Curve for neuron %d', neuron_id);%%% There are 100 x 8 = 800 trials in total
title(str);
xlabel('Direction No');
ylabel('Number of Occurance');

