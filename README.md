# SCALE-Net for Cross-Subject EEG Generalization
This framework, Spectral Convolutional Attention (SE) LSTM Encoder (SCALE-Net), aims to improve the cross subject generalization in EEG signals. This project was part of the CMU Introduction to Deep Learning course. 

## SCALE-Net

To train (options for task name are SSVEP, P300, MI, Imagined_speech, and all):

      cd scale-net
      python -m train_scale_net —task [task name]
      
To test (options for task name are SSVEP, P300, MI, Imagined_speech, and all):

      python -m test_scale_net —task [task name]

## SCALE-Net + EEGNet
This version uses temporal and spectral data. Features are added together in a balanced manner.

To train (options for task name are SSVEP, P300, MI, Imagined_speech, and all):

      python -m train_scalenet_eegnet —task [task name]
      
## Adaptive SCALE-Net
This is our most accurate model. It uses temporal and spectral data with weighted featuring.

To train (options for task name are SSVEP, P300, MI, Imagined_speech, and all):

      python -m scale_net_adaptive —task [task name]
      
