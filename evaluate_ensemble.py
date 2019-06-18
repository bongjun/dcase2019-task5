import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np
import pandas as pd
import oyaml as yaml
from collections import OrderedDict

from urban_sound_tagging_baseline.classify import get_file_targets, get_subset_split, generate_output_file
from urban_sound_tagging_baseline.metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc

import torch
from model_architecture import MyCNN
from train import load_mels

def predict_ensemble(mel_list, test_file_idxs, model1, model2):

	y_pred = []

	with torch.no_grad():
		for idx in test_file_idxs:

			test_x = mel_list[idx]
			test_x = np.reshape(test_x,(1,1,test_x.shape[0],test_x.shape[1]))
			
			if torch.cuda.is_available():
				test_x = torch.from_numpy(test_x).cuda().float()
			else:
				test_x = torch.from_numpy(test_x).float()
			
			model_output = (model1(test_x) + model2(test_x))/2

			if torch.cuda.is_available():
				model_output = model_output.cpu().numpy()[0].tolist()
			else:
				model_output = model_output.numpy()[0].tolist()

			y_pred.append(model_output)

	return y_pred


def evaluate_ensemble(annotation_path, taxonomy_path, mel_dir, models_dir1, 
					models_dir2, output_dir):
	
	os.makedirs(output_dir, exist_ok=True)

	# Load annotations and taxonomy
	print("* Loading dataset.")
	annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
	with open(taxonomy_path, 'r') as f:
		taxonomy = yaml.load(f, Loader=yaml.Loader)

	file_list = annotation_data['audio_filename'].unique().tolist()

	train_file_idxs, test_file_idxs = get_subset_split(annotation_data)


	print('load mel spectrograms')
	mel_list = load_mels(file_list, mel_dir)

	model_list = [f for f in os.listdir(models_dir1) if 'pth' in f]
	val_loss = [float(f.split('_')[-1][:-4]) for f in model_list]
	model_filename = model_list[np.argmin(val_loss)]

	model1 = MyCNN()
	if torch.cuda.is_available():
		model1.load_state_dict(torch.load(os.path.join(models_dir1, model_filename)))
		model1.cuda()
	else:
		model1.load_state_dict(torch.load(os.path.join(models_dir1, model_filename), map_location='cpu'))

	model1.eval()

	model_list = [f for f in os.listdir(models_dir2) if 'pth' in f]
	val_loss = [float(f.split('_')[-1][:-4]) for f in model_list]
	model_filename = model_list[np.argmin(val_loss)]

	model2 = MyCNN()
	if torch.cuda.is_available():
		model2.load_state_dict(torch.load(os.path.join(models_dir2, model_filename)))
		model2.cuda()
	else:
		model2.load_state_dict(torch.load(os.path.join(models_dir2, model_filename), map_location='cpu'))

	model2.eval()

	y_pred = predict_ensemble(mel_list, test_file_idxs, model1, model2)

	aggregation_type = 'max'
	label_mode = 'coarse'
	generate_output_file(y_pred, test_file_idxs, output_dir, file_list,
						 aggregation_type, label_mode, taxonomy)


	mode = 'coarse'
	prediction_path = os.path.join(output_dir, 'output_max.csv')
	df_dict = evaluate(prediction_path,
						   annotation_path,
						   taxonomy_path,
						   mode)

	micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
	macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)
	
	# Get index of first threshold that is at least 0.5
	thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).nonzero()[0][0]

	print("{} level evaluation:".format(mode.capitalize()))
	print("======================")
	print(" * Micro AUPRC:           {}".format(micro_auprc))
	print(" * Micro F1-score (@0.5): {}".format(eval_df["F"][thresh_0pt5_idx]))
	print(" * Macro AUPRC:           {}".format(macro_auprc))
	print(" * Coarse Tag AUPRC:")

	for coarse_id, auprc in class_auprc.items():
		print("      - {}: {}".format(coarse_id, auprc))




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("annotation_path")
	parser.add_argument("taxonomy_path")
	parser.add_argument("mel_dir", type=str)
	parser.add_argument("models_dir1", type=str)
	parser.add_argument("models_dir2", type=str)
	parser.add_argument("output_dir", type=str)

	args = parser.parse_args()

	evaluate_ensemble(args.annotation_path, 
					args.taxonomy_path,
					args.mel_dir,
					args.models_dir1, 
					args.models_dir2,
					args.output_dir)