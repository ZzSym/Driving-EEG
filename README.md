# Driving-EEG

## Introduction

This is a repository about decoding continuous EEG of driving tasks, including preprocessing, ML analysis and DL modeling methods

The data I use is multimodal data collected in a virtual driving environment, including EEG with 63 channels, steering wheel angle, location and speed information, etc. This repository is only a preliminary study on continuous EEG decoding. The purpose is to try the data fitting effects of various commonly used models. Due to the high signal-to-noise ratio of EEG, the results of these models may not be ideal, but these methods and ideas are very important in the data analysis process of BCI. Hope this repository is helpful to you.

Please note that if you want to analyze the model, methods such as k-fold cross-validation should be used (in this repository, k-fold cross-validation is not used for the purpose of demonstrating the method). Due to the differences in models, the repository does not perform a unified classification or regression task. Please be flexible and adaptable according to your question.

## Todo list

### *Data Preprocessing*

- [x] EEG preproceesing
- [x] data alignment

### *Classical Machine Learning*

- [x] LDA
- [x] logistic regression
- [x] SVM

### *Neural Networks*

- [] CNN
- [] RNN
- [] LSTM
- [] MLP
- [] Transformer

### *Foundation Models*

- [] LaBraM
- [] EEGPT

### *Others*

- [] Contrastive Learning
