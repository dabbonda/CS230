import numpy as np
import sys
import imgaug as ia
from imgaug import augmenters as iaa
import os
np.set_printoptions(threshold=np.nan)




def Augment(images, file_names):
    # images = np.array(images)
    images = np.rollaxis(np.dstack(images), -1)
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
                rotate=(-15, 15),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
            )),
        ],
        random_order=True
    )

    images_aug = seq.augment_images(images)
    # np.save(file_name, images_aug[0])

    count = 0
    for i in images_aug:
        np.save("aug-" + file_names[count], i)
        count+=1



images = []
file_names = []

images_256 = []
file_names_256 = []
array_files = os.listdir(os.getcwd())
for array in array_files:
    try:
        image = np.load(array)
    except:
        print array
        continue
    if image.shape == (512, 512):
        images.append(image)
        file_names.append(array)
    elif image.shape == (256, 256):
        images_256.append(image)
        file_names_256.append(array)

Augment(images, file_names)
Augment(images_256, file_names_256)


