import librosa
import os
import numpy as np
from vggish_utils import vggish_input_clipwise
import argparse

def wav_to_mel(filename):
	y, sr = librosa.load(filename, mono=True, sr=None)
	if y.shape[0]<sr*1 and y.shape[0]>sr*0.0:
		y=librosa.util.fix_length(y, int(sr*1.01))
	y = y.T
	
	mel = vggish_input_clipwise.waveform_to_examples(y, sr)

	return mel

def extract_mel(dataset_dir, output_dir):
	os.makedirs(output_dir, exist_ok=True)

	filenames = os.listdir(dataset_dir)
	counter = 0
	for i, filename in enumerate(filenames):
		if 'wav' in filename:
			print(i, filename)
			audio_path = os.path.join(dataset_dir, filename)
			mel = wav_to_mel(audio_path)
			mel_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')
			np.save(mel_path, mel)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir")
	parser.add_argument("output_dir")

	args = parser.parse_args()

	extract_mel(dataset_dir=args.dataset_dir,
				output_dir=args.output_dir)


