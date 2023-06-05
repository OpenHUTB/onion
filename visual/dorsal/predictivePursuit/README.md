# predictivePursuit
Ubuntu下MATLAB编辑器中文乱码的问题：
Windows下.m文件编码不是utf-8,需要转换
安装enca:
sudo apt install enca
在对应文件夹下把所有.m 文件转换过来
enca -L zh_CN -x utf-8 *.m

Code accompanying the paper:

Kashyap, H. J., Detorakis, G., Dutt, N., Krichmar, J. L., & Neftci, E. (2018). A Recurrent Neural Network Based Model of Predictive Smooth Pursuit Eye Movement in Primates. In International Joint Conference on Neural Networks (IJCNN).

http://www.socsci.uci.edu/~jkrichma/Kashyap-PredicitvePursuit-IJCNN2018.pdf

The code is tested using Matlab 2017a. If you use this code in your research, please cite the above paper.

Files (sp: smooth pursuit):

main_sp_predict.m (运行预测和初始化实验的主脚本。 main script to run the prediction and the initiation experiemnts)

sp_predict_ramp_pertur.m (带抖动和相移的预测性跟踪实验脚本。 script to run the predictive pursuit experiment with perturbation and phase shift)
ramp stimulus: 上升刺激

sp_predict_random_target.m (对随机目标的预测性跟踪实验脚本。script to run the predictive pursuit experiemnt for a random target)

get_accelaration.m (计算初始化阶段跟踪的加速。Calculates pursuit accelaration during initiation)

get_RS.m (绘制视网膜滑动。plots retinal slip)

getWeights.m (RNN突触的权重。RNN synapse weights)
