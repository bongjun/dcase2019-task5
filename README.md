# DCASE 2019-task5
This repository contains code to build models for [DCASE 2019 Challange-task5 (Urban sound tagging)](http://dcase.community/challenge2019/task-urban-sound-tagging). One of the submitted models got **3rd** place out of 22 systems competing (coarse-level prediction). Check [the results](http://dcase.community/challenge2019/task-urban-sound-tagging-results) out.

* If you use this code for your research paper, please cite the following technical report: [pdf](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Kim_107.pdf)

```
@techreport{Kim2019,
    Author = "Kim, Bongjun",
    abstract = "This technical report describes sound classification models from our submissions for DCASE challenge 2019-task5. The task is to build a system to perform audio tagging on urban sound. The dataset has 23 fine-grained tags and 8 coarse-grained tags. In this report, we only present a model for coarse-grained tagging. Our model is a Convolutional Neural network (CNN)-based model which consists of 6 convolutional layers and 3 fully-connected layers. We apply transfer learning to the model training by utilizing VGGish model that has been pre-trained on a large scale of a dataset. We also apply an ensemble technique to boost the performance of a single model. We compare the performance of our models and the baseline approach on the provided validation dataset. The results show that our models outperform the baseline system.",
    month = "September",
    year = "2019",
    title = "Convolutional Neural Networks with Transfer Learning for Urban Sound Tagging",
    institution = "DCASE2019 Challenge"
}
```


* Some scripts for manipulating the dataset and evaluating the model were taken from [the baseline code](https://github.com/sonyc-project/urban-sound-tagging-baseline) provided by the challenge organizers.
* The feature extraction codes (`vggish_utils/`) are from the repository of [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset).

## Task description
The goal is to predict whether each of 23 (fine-grained) or 8 (coarse grained) sources of noise pollution is present in a 10-second recording. This is a multi-label and multi-class classification problem. **My model only works for the 8 coarse-grained labels:**
* Classes: engine, machinery-impact, non-machinery-impact, powered-saw, alert-signal, music, human-voice, and dog.

## Model description
* a CNN-based model: 6 convolutional layers +  3 fully-connected layers
* Transfer learning from a part of [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset)
* It takes a mel-spectrogram of 10-second recording and outputs scores which can be interpreted as probabilities of each class being present in the recording.
* While data augmentation and using external datasets are allowed in this task, none of them was used in this submission. After the labels for evaluation set is released, I will apply simple data augmentation techniques just to see if they help.

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

* Install python packages in your virtual environment (or conda):
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
Firat, extract mel-spectrograms of audio files in `data/train` and `data/validate` directories of the challenge dataset.

```shell
python extract_mel.py data/annotations-dev.csv data mels
```
This will create `mels` directory and store a set of melspectrogram of training and valudation files in the directory.

## Training the first model (model#1)
```shell
python train.py data/annotations-dev.csv data/dcase-ust-taxonomy.yaml mels checkpoints validation_output
```
It will create `checkpoints` and `validation_output` directories. After the training is done, you will see a set of model checkpoints in `checkpoints` and a result csv file (`output_max.csv `) in the `validation_output` directory.

## Evaluating model#1 on the validation set.
```shell
python evaluate.py data/annotations-dev.csv validation_output/output_max.csv data/dcase-ust-taxonomy.yaml
```

It will report the performance of the best model on the validation set.

## Generating the submission file for model#1

#### Extracting mel-spectrograms of the evaluation set.

```shell
python eval_extract_mel.py data/audio-eval eval_mels
```

This will create `eval_mels` directory and store mel-spectrograms of all audio files in `data/audio-eval` directory.

#### Testing your model and generate the submission file.
```shell
python gen_submission.py data/annotations-dev.csv data/dcase-ust-taxonomy.yaml eval_mels checkpoints submision_file1
```
This will create `submision_file` directory and store `output_max.csv` in the directory.

Now you've just trained and tested the first model. You can build the second and the thrid models in the similar way.

## Build the second model (model#2)
#### Training
```shell
python train.py data/annotations-dev.csv data/dcase-ust-taxonomy.yaml mels checkpoints2 validation_output2 --learning_rate=1e-3
```
It will create `checkpoints2` and `validation_output2` directories. After the training is done, you will see a set of model checkpoints in `checkpoints2` and a result csv file (`output_max.csv `) in the `validation_output2` directory.

#### Evaluating on validation set
```shell
python evaluate.py data/annotations-dev.csv validation_output2/output_max.csv data/dcase-ust-taxonomy.yaml
```
It will report the performance of the best model on the validation set.

#### Generate the submission file for the model2

```shell
python gen_submission.py data/annotations-dev.csv data/dcase-ust-taxonomy.yaml eval_mels checkpoints2 submision_file2
```
This will create `submision_file2` directory and store `output_max.csv` in the directory.

## The ensemble model (model#1 + model#2)
#### Evaluating on validation set
```shell
python evaluate_ensemble.py data/annotations-dev.csv data/dcase-ust-taxonomy.yaml mels checkpoints checkpoints2 validation_output_ensemble
```
It will report the performance of the ensemble model on the validation set.

#### Generate the submission file for the ensemble model (model3)
```shell
python gen_submission_ensemble.py data/annotations-dev.csv data/dcase-ust-taxonomy.yaml eval_mels checkpoints checkpoints2 submision_file3
```
This will create `submision_file3` directory and store `output_max.csv` in the directory.

