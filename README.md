# DCASE 2019-task5
This repository contains code to build models for [DCASE 2019 Challange-task5 (Urban sound tagging)](http://dcase.community/challenge2019/task-urban-sound-tagging).
* Some scripts for manipulating the dataset and evaluating the model were taken from [the baseline code](https://github.com/sonyc-project/urban-sound-tagging-baseline) provided by the challange orginzers.
* The feature extraction codes (`vggish_utils/`) are from the repository of [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset).

## Task description
The goal is to predict whether each of 23 (fine-grained) or 8 (coarse grained) sources of noise pollution is present in a 10-second recording. This is a multi-label and multi-class classification probelem. **My model only works for the 8 coarse-grained labels:**
* Classes: engine, machinery-impact, non-machinery-impact, powered-saw, alert-signal, music, human-voice, and dog.

## Model description
* a CNN-based model: 6 convolutional layers +  3 fully-connceted layers
* Transfer learning from a part of [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset)
* It takes a mel-spectrogram of 10-second recording and outputs scores which can be interpreted as probabilities of each class being present in the recording.
* While data augmentation and using exteranl datasets are allowed in this task, none of them was used in this submission. After the labels for evaluation set is released, I will apply simple data augmentation techinques just to see if they help.

## Installation
You will need `Python3` to run the codes.

* Clone this repository and enter it.

```shell
git clone https://github.com/bongjun/dcase2019-task5.git
cd dcase2019-task5
```
* Set up your Python virtual environment.

```
virtualenv --system-site-packages -p python3 ./dcase_venv
source ./dcase_venv/bin/activate
```

* Install python packages in your virtual envorinment (or conda):
```shell
pip install -r requirements.txt
```

* Install PyTorch

Find the correct install command for your operating system and version of python [here](https://pytorch.org/)

* Download the dataset and extracting the files.
```shell
mkdir -p data
pushd data
wget https://zenodo.org/record/3233082/files/annotations-dev.csv
wget https://zenodo.org/record/3233082/files/audio-dev.tar.gz
wget https://zenodo.org/record/3233082/files/dcase-ust-taxonomy.yaml
wget https://zenodo.org/record/3233082/files/audio-eval.tar.gz
tar -xvzf audio-dev.tar.gz
tar -xvzf audio-eval.tar.gz
popd
```

## Preparation
Firat, extract mel-spectrograms of audio files in `train` and `validate` directories of the challenge dataset. Download [the challenge dataset](https://zenodo.org/record/3233082#.XQKIRW9KiL4) and unzip `audio-dev.tar.gz` in your project directory.

```shell
python extract_mel.py YOUR_PROJECT_DIRECTORY/annotations.csv YOUR_PROJECT_DIRECTORY/data YOUR_PROJECT_DIRECTORY/mels
```
This will create `mels` directory and store a set of melspectrogram of training and valudation files in the directory.

## Training
```shell
python train.py YOUR_PROJECT_DIRECTORY/annotations.csv YOUR_PROJECT_DIRECTORY/urban_sound_tagging_baseline/dcase-ust-taxonomy.yaml YOUR_PROJECT_DIRECTORY/mels YOUR_PROJECT_DIRECTORY/checkpoints YOUR_PROJECT_DIRECTORY/validation_output
```
It will create `checkpoints` and `validation_output` directories. After the training is done, you will see a set of model checkpoints in `checkpoints` and a result csv file (`output_max.csv `) in the `validation_output` directory.

## Evaluating models on the validation set.
```shell
python evaluate.py YOUR_PROJECT_DIRECTORY/annotations.csv YOUR_PROJECT_DIRECTORY/validation_output/output_max.csv YOUR_PROJECT_DIRECTORY/urban_sound_tagging_baseline/dcase-ust-taxonomy.yaml
```

It will report the performance of the best model on the validation set.

## Generating the submission file

#### Extracting mel-spectrograms of the evaluation set.
Download [the challenge dataset](https://zenodo.org/record/3233082#.XQKIRW9KiL4) and unzip `audio-eval.tar.gz` in your project directory.

```shell
python eval_extract_mel.py YOUR_PROJECT_DIRECTORY/audio-eval YOUR_PROJECT_DIRECTORY/eval_mels
```

This will create `eval_mels` directory and store mel-spectrograms of all audio files in `audio-eval` directory.

#### Testing your model and generate the submission file.
```shell
python gen_submission.py YOUR_PROJECT_DIRECTORY/annotations.csv YOUR_PROJECT_DIRECTORY/urban_sound_tagging_baseline/dcase-ust-taxonomy.yaml YOUR_PROJECT_DIRECTORY/eval_mels/ YOUR_PROJECT_DIRECTORY/checkpoints/ YOUR_PROJECT_DIRECTORY/submision_file/
```
This will create `submision_file` directory and store `output_max.csv` in the directory.



