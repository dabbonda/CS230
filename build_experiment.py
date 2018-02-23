"""Split the FETAL dataset into train/val/test sets.

The FETAL dataset comes into the following format:
FETAL/
processed/
0-0002.npy
...

The output .npy files are placed in the following directory:
FETAL/
train/
0-0002.npy
...
test/
0-0002.npy
...
val/
0-0002.npy
...        

Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take a 70%-15%-15% split into train-val-test sets.
"""

import sys
import os
import random
import os
import numpy as np

from PIL import Image
from tqdm import tqdm

# directory pointing to all .npy files
input_directory = './data/FETAL/processed'

# experiment directories
output_directory = './data/FETAL'

def slice_and_save(filename, output_dir):
	"""Slice the 3d numpy array contained in `filename` into 2d numpy arrays and save 
	it to the `output_dir`"""
	raw_matrix = np.load(filename)
# print(filename)
# print(output_dir)
# print()
	= os.path.split(filename)[1].split(".")[0]
	for slice_num, raw_slice in enumerate(raw_matrix):
	output_file_name = "%s_%s.npy" % (input_file, slice_num)
	output_file_path = os.path.join(input_fileoutput_dir, output_file_name)
	np.save(output_file_path, raw_slice)

	if __name__ == '__main__':

	assert os.path.isdir(input_directory), "Couldn't find the dataset at {}".format(input_directory)

# Get the filenames in the input directory
filenames = os.listdir(input_directory)
	filenames = [os.path.join(input_directory, f) for f in filenames if f.endswith('.npy')]

# Split the image into 70% train and 15% val and 15% test
# Make sure to always shuffle with a fixed seed so that the split is reproducible
	random.seed(230)
	filenames.sort()
random.shuffle(filenames)

	first_split = int(0.70 * len(filenames))
second_split = int(0.85 * len(filenames))
	train_filenames = filenames[:first_split]
	val_filenames = filenames[first_split:]
	test_filenames = filenames[second_split:]

	filenames = {'train': train_filenames,
		'val': val_filenames,
		'test': test_filenames}

		if not os.path.exists(output_directory):
			os.mkdir(output_directory)
			else:
				print("Warning: output dir {} already exists".format(output_directory))

# Preprocess train, val and test
				for split in ['train', 'val', 'test']:
				output_dir_split = os.path.join(output_directory, '{}'.format(split))
				if not os.path.exists(output_dir_split):
					os.mkdir(output_dir_split)
					else:
						print("Warning: dir {} already exists".format(output_dir_split))

						print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
						for filename in tqdm(filenames[split]):
							slice_and_save(filename, output_dir_split)

							print("Done building dataset")
