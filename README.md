# Speech-Experiment
Multi purpose speech experiment platform: speech separation, speech enhancement, speaker recognition
TIMIT&MUSAN -free sound is adopted. The data mixing adopts completely random selection, which can potentially improve the robustness of the model.

# Speech Separation:
Embedding 2 mask: DANET, DPCL is complete
PIT: TasNet (R_latten), ConvTasNet, DPRNN, Chimira++

# Speech Enhancememt
similar to PIT， DarkConvTasNet, DPT, Sudormrf

# Classfier:
ResNet18, 32, MobilenetV3, Nonuttrance classfier， ResNet34, DenseNet all version!!
