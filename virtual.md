# 虚拟人预测驾驶行为的神经激活结构

图1 使用模仿在驾驶行为上比较人驾驶和虚拟人驾驶（行为比较: 代理的驾驶行为 vs 人的驾驶行为） 控制行为如何比较相似性

图2 使用模仿训练人工代理来模拟人的驾驶行为（行为预测）

图3 通过逆动力学模型能最好地预测DLS和 MC的神经激活（激活预测）控制模型充当逆动力学模型

图4 DLS和MC中神经群体在驾驶行为上的代表性结构类似于逆模型（激活 代表 行为，逆模型）

图5 随机控制器通过改变潜在的可变性来调节运动可变性作为行为的功能（改变 隐控制器 能 改变行为）


* 补充视频1 
模拟管道概述。模拟管道由多摄像机视频采集组成。

* 补充视频2
使用DANNCE进行精确的3D姿势估计。我们使用DANNCE从多摄像机记录中估计自由移动的老鼠的3D姿势。这段视频描述了覆盖在所有六个摄像头原始视频记录之上的DANNCE keypoint估计值。关键点估计在各种行为中都是准确的。

## 类脑驾驶网络

进化塑造了结构
结构 影响激活，自激活（无意识+意识（范围变化））+外部输入的刺激
激活
行为


## 人在驾驶舱
建模刹车、方向盘、档位
Carla中行人接受声音，客户端目前不支持接受服务端声音，可以在Carla汽车模型中添加引擎声音。
服务端运行在windows，客户端可以运行在WSL中。



## 参考

[虚拟鼠预测行为的神经激活结构](https://github.com/diegoaldarondo/virtual_rodent)


[多相机视频获取](https://github.com/ksseverson57/campy)

[从多个相机估计3D姿态](https://github.com/spoonsso/dannce)

[骨架注册](https://github.com/diegoaldarondo/stac/tree/6cd5d05170cff993b26949e35e2b17e58568fbc2)

[逆动力学模型训练](https://github.com/google-deepmind/dm_control/tree/main/dm_control/locomotion)

[逆动力学模型推断](https://github.com/diegoaldarondo/npmp_embedding/tree/788baf281a613322821b07e06413648b0f7a1b79)

[行为分类](https://github.com/gordonberman/MotionMapper) 

脉冲尖峰排序

表征相似性分析[Python](https://github.com/rsagroup/rsatoolbox) 

[数据分析库PyData](https://pydata.org/)


### 其他






[老鼠跳舞的捕捉](https://github.com/jessedmarshall/CAPTURE_demo)

[多动物的姿势跟踪](https://github.com/talmolab/sleap)

[动作序列分析](https://github.com/dattalab/keypoint-moseq)

[运动技能学习](https://github.com/lucylai96/motor-rl) 

