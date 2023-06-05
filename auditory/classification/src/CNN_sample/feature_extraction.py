'''
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
'''
import os
import numpy as np
import librosa

from hparams import hparams

def load_list(list_name, hparams):
	with open(os.path.join(hparams.dataset_path, list_name)) as f:
		file_names = f.read().splitlines()
	return file_names

def load_data(file_name, hparams):
	y, sr = librosa.load(os.path.join(hparams.dataset_path, file_name), hparams.sample_rate)
	# S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)
	#
	# mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
	# mel_S = np.dot(mel_basis, np.abs(S))
	# mel_S = np.log10(1+10*mel_S)
	# mel_S = mel_S.T
	return y

def resize_array(array, length):
	resized_array = np.zeros((length,1))
	num_array = np.expand_dims(array, axis=1)
	if num_array.shape[0] >= length:
		resized_array = num_array[:length]
	else:
		resized_array[:num_array.shape[0]] = num_array[:num_array.shape[0]]

	print(resized_array.shape)
	return resized_array

def main():
	print('Extracting Feature')
	list_names = ['train_list.txt', 'valid_list.txt', 'test_list.txt']
	for list_name in list_names:
		set_name = list_name.replace('_list.txt', '')
		file_names = load_list(list_name, hparams)

		for file_name in file_names:
			data = load_data(file_name, hparams)
			data = resize_array(data, hparams.max_length)

			data_chuck = np.split(data, 10)

			for idx, i in enumerate(data_chuck):
				save_path = os.path.join(hparams.feature_path, set_name, file_name.split('/')[0])
				save_name = file_name.split('/')[1].split('.wav')[0] + str(idx) + ".npy"
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				np.save(os.path.join(save_path, save_name), i.astype(np.float32))
				print(os.path.join(save_path, save_name))

	print('finished')

if __name__ == '__main__':
	main()