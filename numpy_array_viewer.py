import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def display_image(file_name, is3D):
    data = np.load(file_name)
    if is3D:
        sliceBitmap = data[5]  # random slice of the mri
    else:
        sliceBitmap = data
    plt.imshow(sliceBitmap, cmap=cm.Greys_r)
    plt.show()