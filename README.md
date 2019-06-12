(To be updated soon).
# dcase2019-task5
This repository contains code to build models for [DCASE 2019 Challange-task5 (Urban sound tagging)](http://dcase.community/challenge2019/task-urban-sound-tagging). Some scripts for manipulating the dataset and evaluating the model were taken from [the baseline code](https://github.com/sonyc-project/urban-sound-tagging-baseline) provided by the challange orginzers.

## The summary of the task
The goal is to predict whether each of 23 (fine-grained) or 8 (coarse grained) sources of noise pollution is present in a 10-second recording. This is a multi-label and multi-class classification probelem. My model only works for the 8 coarse-grained labels:
* Classes: engine, machinery-impact, non-machinery-impact, powered-saw, alert-signal, music, human-voice, and dog.

## The summary of the model
* a CNN-based model: 6 convolutional layers +  3 fully-connceted layers
* Transfer learning from a part of [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset)
* It takes a mel-spectrogram of 10-second recording and outputs scores which can be interpreted as probabilities of each class being present in the recording.

## Installation

## Preparation

## Training

## Testing

