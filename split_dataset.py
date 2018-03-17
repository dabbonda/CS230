"""Split the FETAL dataset into train/val/test sets.

The FETAL dataset comes into the following format:
FETAL/
    processed/
        0-0001.npy
        ...

The output .npy files are placed in the following directory:
FETAL/
    train/
        0-0001.npy
        ...
    test/
        0-0002.npy
        ...
    val/
        0-0003.npy
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

# whether to use smaller training set
USE_SMALL_DATA = False
SMALL_DATA_CUTOFF = 0.05

# split into 3D numpy arrays flag
SPLIT_3D = True

# training-val-dev split
TRAIN_CUTOFF = 0.70
VAL_CUTOFF = 0.85

# directory pointing to all .npy files
input_directory = './data/FETAL/processed'

# experiment directories
output_directory = './data/FETAL/3d'

# takes the array right before it is saved, divides by max value multiplies by 255 and rounds to an int array
def normalize_255(a):
    a = a/float(np.max(a))
    a = a * 255
    a = np.rint(a).astype(int)
    return a


def slice_and_save(filename, output_dir):
    """Slice the 3d numpy array contained in `filename` into 2d numpy arrays and save 
    it to the `output_dir`"""
    raw_matrix = np.load(filename)
    # print(filename)
    # print(output_dir)
    # print()
    input_file = os.path.split(filename)[1].split(".")[0]
    if SPLIT_3D:
        output_file_name = "%s.npy" % (input_file)
        output_file_path = os.path.join(output_dir, output_file_name)
        np.save(output_file_path, normalize_255(raw_matrix))
        os.remove(filename)
    else:
        for slice_num, raw_slice in enumerate(raw_matrix):
            output_file_name = "%s_%s.npy" % (input_file, str(slice_num).zfill(4))
            output_file_path = os.path.join(output_dir, output_file_name)
            np.save(output_file_path, normalize_255(raw_slice))


if __name__ == '__main__':

    assert os.path.isdir(input_directory), "Couldn't find the dataset at {}".format(input_directory)

    # Get the filenames in the input directory
    filenames = os.listdir(input_directory)
    filenames = [os.path.join(input_directory, f) for f in filenames if f.endswith('.npy')]

    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    # Whether to use a smaller subset
    if USE_SMALL_DATA:
        small_split = int(SMALL_DATA_CUTOFF * len(filenames))
        filenames = filenames[:small_split]

        # Reshuffle the filenames
        filenames.sort()
        random.shuffle(filenames)

    # Split the image into 70% train and 15% val and 15% test
    first_split = int(TRAIN_CUTOFF * len(filenames))
    second_split = int(VAL_CUTOFF * len(filenames))
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
