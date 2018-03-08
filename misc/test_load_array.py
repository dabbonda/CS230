import random
import numpy as np
import glob, os, sys, shutil, dicom

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# raw_image = np.load("data/FETAL/val/0_0002_7.npy")    # load numpy array from .npy file
# raw_image = raw_image * (255.0 / raw_image.max())
# image = Image.fromarray(raw_image)          # PIL image
# image.show()

abnormal_count = 0
abnormal_count += len(glob.glob("%s/0*" % "./data/FETAL/train"))
abnormal_count += len(glob.glob("%s/0*" % "./data/FETAL/val"))
abnormal_count += len(glob.glob("%s/0*" % "./data/FETAL/test"))

print("Abnormal Count: ", abnormal_count)

normal_count = 0
normal_count += len(glob.glob("%s/1*" % "./data/FETAL/train"))
normal_count += len(glob.glob("%s/1*" % "./data/FETAL/val"))
normal_count += len(glob.glob("%s/1*" % "./data/FETAL/test"))

print("Normal Count: ", normal_count)

print("Percent Normal: ", 100.0 * float(normal_count) / (normal_count + abnormal_count))