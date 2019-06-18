import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np
import pandas as pd
import oyaml as yaml

from urban_sound_tagging_baseline.classify import get_file_targets, get_subset_split, generate_output_file
from urban_sound_tagging_baseline.metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from model_architecture import MyCNN

from train import load_mels, predict
from evaluate_ensemble import predict_ensemble

def evaluation(annotation_path, taxonomy_path, mel_dir, 
					models_dir1, models_dir2, output_dir):
	
	os.makedirs(output_dir, exist_ok=True)

	with open(taxonomy_path, 'r') as f:
		taxonomy = yaml.load(f, Loader=yaml.Loader)

	file_list = [os.path.splitext(f)[0]+'.wav' for f in os.listdir(mel_dir) if 'npy' in f]
	file_list.sort()

	test_file_idxs = range(len(file_list))

	model_list = [f for f in os.listdir(models_dir1) if 'pth' in f]
	val_loss = [float(f.split('_')[-1][:-4]) for f in model_list]
	model_filename = model_list[np.argmin(val_loss)]

	model1 = MyCNN()
	model1.load_state_dict(torch.load(os.path.join(models_dir1, model_filename)))
	if torch.cuda.is_available():
		model1.cuda()
	model1.eval()

	model_list = [f for f in os.listdir(models_dir2) if 'pth' in f]
	val_loss = [float(f.split('_')[-1][:-4]) for f in model_list]
	model_filename = model_list[np.argmin(val_loss)]

	model2 = MyCNN()
	model2.load_state_dict(torch.load(os.path.join(models_dir2, model_filename)))
	if torch.cuda.is_available():
		model2.cuda()
	model2.eval()

	mel_list = load_mels(file_list, mel_dir)
	y_pred = predict_ensemble(mel_list, test_file_idxs, model1, model2)

	aggregation_type = 'max'
	label_mode='coarse'
	generate_output_file(y_pred, test_file_idxs, output_dir, file_list,
								 aggregation_type, label_mode, taxonomy)


	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("annotation_path", type=str)
	parser.add_argument("taxonomy_path", type=str)
	parser.add_argument("mel_dir", type=str)
	parser.add_argument("models_dir1", type=str)
	parser.add_argument("models_dir2", type=str)
	parser.add_argument("output_dir", type=str)

	args = parser.parse_args()

	evaluation(args.annotation_path,
		args.taxonomy_path,
		args.mel_dir,
		args.models_dir1,
		args.models_dir2,
		args.output_dir)
