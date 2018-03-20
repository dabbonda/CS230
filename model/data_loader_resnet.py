import random
import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    #transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip()])
# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    #transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.Grayscale(num_output_channels=3)])
# Original images have size (512, 512) or (256, 256). Resizing to (64, 64) reduces the dataset size, 
# and loading smaller images makes training faster.

class FETALDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.npy')]

        self.labels = [int(filename.split('/')[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        raw_image = np.load(self.filenames[idx])    # load numpy array from .npy file
        raw_image = raw_image * (255.0 / raw_image.max()) if raw_image.max() != 0 else raw_image
        image3d=[]
        for i in np.arange(raw_image.shape[0]):
        #for i in np.arange(10):
            image = Image.fromarray(raw_image[i,:,:])          # PIL image
            image = image.resize((224, 224), Image.BILINEAR)
            image = self.transform(image) 
            image = transforms.ToTensor()(image)
            image3d.append(image)
        image=torch.stack(image3d, -1)
        
        
        
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(FETALDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(FETALDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
