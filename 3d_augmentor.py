import numpy as np
import sys
import imgaug as ia
from imgaug import augmenters as iaa
import os

np.set_printoptions(threshold=np.nan)


def Augment(image, file_name):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images

            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
            )),
        ],
        random_order=True
    )

    image_aug = seq.augment_image(image)
    # np.save(file_name, images_aug[0])

    np.save(file_name, image_aug)


array_files = os.listdir(os.getcwd())
for array in array_files:
    try:
        image = np.load(array)
        image = np.rollaxis(image, 1)
        image = np.rollaxis(image, 2)
        print image.shape
    except:
        print array + " could not be opened"
        continue
    Augment(image, "aug-" + array)
