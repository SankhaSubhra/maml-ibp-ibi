import os
import random

from PIL import Image
import numpy as np
import torch

def read_dataset(data_dir, val_presence=True):

    # Read the image dataset.

    if val_presence is True:
        return tuple(_read_classes(os.path.join(data_dir, x)) for x in ['train', 'val', 'test'])
    else:
        return tuple(_read_classes(os.path.join(data_dir, x)) for x in ['train', 'test', 'test'])

def _read_classes(dir_path):
    
    # Read the class directories in a train/val/test directory.
    # Images should be in ".jpg" format.
    
    return [ImageProcessClass(os.path.join(dir_path, f)) for f in os.listdir(dir_path)]

class ImageProcessClass:
    
    # Loading and using the image dataset.
    # To use these APIs, you should prepare a directory that
    # contains three sub-directories: train, test, and val.
    
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self._cache = {}

    def sample(self, num_images):

        # Sample images (as pytorch tensor) from the class.

        names = [f for f in os.listdir(self.dir_path) if f.endswith('.jpg')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(name))
        return images

    def _read_image(self, name):

        # For reading images and transformations as necessary.
        # Image resolution is set to 84x84.

        if name in self._cache:
            return self._cache[name]
        with open(os.path.join(self.dir_path, name), 'rb') as in_file:
            img = Image.open(in_file).resize((84, 84)).convert('RGB')
            img = np.array(img).astype('float32') / 0xff
            img =np.rollaxis(img, 2, 0)
            self._cache[name] = torch.tensor(img)
            return self._read_image(name)
