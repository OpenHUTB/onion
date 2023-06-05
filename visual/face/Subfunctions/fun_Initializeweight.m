function [net_test, lim, Weights, Biases] = fun_Initializeweight(net, ver, stdfac)
% 随机初始化 AlexNet 的 5 个卷积层（其他层不做处理）

net_test = net;
net_tmp = net_test.saveobj;  % saveobj: 修改对象的保存过程

rand_layers_ind = [2, 6, 10, 12 14];        % 5 个卷积层
Weights = cell(1, length(rand_layers_ind));
Biases = cell(1, length(rand_layers_ind));
for ind_tl = 1:length(rand_layers_ind)
    % ind_tl = 1;
    % LOI = layers_set{ind_tl};
    targetlayer_ind = rand_layers_ind(ind_tl);
    weight_conv = net.Layers(targetlayer_ind, 1).Weights;  % 当前卷积层的权重
    bias_conv = net.Layers(targetlayer_ind, 1).Bias;       % 当前卷积层的偏置
    
    fan_in = size(weight_conv,1) * size(weight_conv,2) * size(weight_conv,3);  % 11*11*3
    
    % (11*11*3*96,  1*1*96 ), 
    % (5*5*48*256,  1*1*256), 
    % (3*3*256*384, 1*1*384), 
    % (3*3*192*384, 1*1*384), 
    % (3*3*192*256, 1*1*256)
    if ver == 1 
        lim(ind_tl) = sqrt(2/fan_in);
        % 在初始化阶段让前向传播过程每层方差保持不变,权重从 N(0, 1/fan_in) 的高斯分布采样 (Efficient
        % BackProp )
        Wtmp = stdfac*randn(size(weight_conv))*sqrt(1/fan_in); % 正态分布（randn） LeCun initializaation
        Btmp = randn(size(bias_conv));
    elseif ver == 2  
        lim(ind_tl) = sqrt(3/fan_in);
        Wtmp = stdfac*(rand(size(weight_conv))-0.5)*2*sqrt(3/fan_in); % 均匀分布（rand） Lecun uniform initializaation
        Btmp = randn(size(bias_conv));
    end

    %% 改变网络参数 change network parameters
    weight_conv_randomize = single(1*Wtmp);
    bias_conv_randomize = single(0*Btmp);
    
    Weights{ind_tl} = weight_conv_randomize;
    Biases{ind_tl} = bias_conv_randomize;
    
    net_tmp.Layers(targetlayer_ind).Weights = weight_conv_randomize;
    net_tmp.Layers(targetlayer_ind).Bias = bias_conv_randomize;
end
net_test = net_test.loadobj(net_tmp);  % loadobj: 自定义对象的加载过程
end