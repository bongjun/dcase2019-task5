import librosa
import os
import numpy as np
from vggish_utils import vggish_input_clipwise
import pandas as pd
import argparse

def wav_to_mel(filename):
	y, sr = librosa.load(filename, mono=True, sr=None)
	if y.shape[0]<sr*1 and y.shape[0]>sr*0.0:
		y=librosa.util.fix_length(y, int(sr*1.01))
	y = y.T
	
	mel = vggish_input_clipwise.waveform_to_examples(y, sr)

	return mel

def extract_mel(annotation_path, dataset_dir, output_dir):
	print("* Loading annotations.")
	annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

	os.makedirs(output_dir, exist_ok=True)

	df = annotation_data[['split', 'audio_filename']].drop_duplicates()

	counter = 0
	for index, row in df.iterrows():
		filename = row['audio_filename']
		print('({}/{}) {}'.format(counter, len(df), filename))
		partition = row['split']
		audio_path = os.path.join(dataset_dir, partition, filename)
		mel = wav_to_mel(audio_path)
		mel_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')
		np.save(mel_path, mel)
		counter+=1



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("annotation_path")
	parser.add_argument("dataset_dir")
	parser.add_argument("output_dir")

	args = parser.parse_args()

	extract_mel(annotation_path=args.annotation_path,
				dataset_dir=args.dataset_dir,
				output_dir=args.output_dir)


