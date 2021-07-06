import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import os


class EncoderDataLoader:
    def __init__(self, params):
        self.data_dir = os.path.join('data_images', params['env'])
        self.frame_stack = params['frame_stack']
        self.env = params['env']
        self.n_images = len(os.listdir(self.data_dir)) // self.frame_stack
        if params['env'] == 'Robot':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.RandomRotation(20),
                transforms.ToTensor
            ])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        ])

    def sample(self, batch_size):
        idxs = np.random.randint(self.n_images, size=batch_size)
        ims = []
        if self.frame_stack == 1:
            template = os.path.join(self.data_dir, '%d.png')
        else:
            template = os.path.join(self.data_dir, '%d_%d.png')
        for idx in idxs:
            if self.frame_stack == 1:
                im = Image.open(template % idx)
                im = self.transform(im)
                ims.append(im)
            else:
                stack = []
                for i in range(self.frame_stack):
                    im = Image.open(template % (idx, i))
                    stack.append(im)
                ims.append(stack)
        return ims


