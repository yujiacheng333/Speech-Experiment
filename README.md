# Speech-Experiment
整合了说话人识别和语音分离的数据集预处理，模型加载交互（基于TIMIT数据集）
还没完工，慢慢完善。
分离Models：DPCL，Conv-TasNet（原版），DPRNN(原版)，DPCL（修改版，没有使用D矩阵），Chimera++（用Non-local替代了BiLSTM，DPCL loss没有添加）
说话人识别Models：简单卷积网络1D，2D ， MobilenetV3原版参数，但是是1D的+LMloss： CMface-loss，稍微改改就能变成其他的face。

# Speech Separation:
Embedding 2 mask: DANET, DPCL is complete
PIT: ConvTasNet, DPRNN, Chimira++

# Speech Enhancememt
similar to PIT

# Classfier:
ResNet18, 32, MobilenetV3, Nonuttrance classfier!!
