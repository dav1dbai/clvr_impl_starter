import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sprites_datagen.moving_sprites import *
from sprites_datagen.rewards import *
from sprites_datagen.general_utils import *
import cv2
import time

# generate images for encoder
class ImgDataSet(Dataset):
    def __init__(self, sprites_ds, len):
        super(ImgDataSet,self).__init__()
        self.spritedataset = sprites_ds
        self.len = len #loader wants length function

    def __getitem__(self, i):
        # return datasets generated image and associated reward
        # print(type(self.spritedataset[0].images))
        return self.spritedataset[0].images, self.spritedataset[0].rewards
    
    def __len__(self):
        return self.len

def dataloader(spec, len, bs):
    # generate new dataset following same specs
    dataset = MovingSpriteDataset(spec)
    imgs = ImgDataSet(dataset, len)
    # create torch dataloader for easier parsing
    loader = DataLoader(imgs, batch_size=bs, shuffle=True)
    return loader

# may have to imlpement diff dataloader for task 2 > one distractor encoder stuff

if __name__ == '__main__':
    # following paper specs
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=4,      # number of shapes per trajectory
        rewards=[ZeroReward],
    )
    loader = dataloader(spec, bs=20, len=1)
    '''
    for b, d in loader:
        print(b.shape) #images
        print(d['zero'].shape) #reward
    '''
    for b,d in loader:
        # returns batch of sequences, returns dict of rewards
        for seq in b:
            print(seq.shape) # sequence of images
            for img in seq:
                print(f'img max/min{img.max()},{img.min()}')
                img += 1
                print(f'img max/min{img.max()},{img.min()}')
                # plt.imshow(img.permute(1,2,0))
                # plt.show()
                # time.sleep(3)
    '''
    sprites_dataset = MovingSpriteDataset(spec)
    traj = TemplateMovingSpritesGenerator(spec).gen_trajectory()
    training_images = ImgDataSet(sprites_dataset)
    print(training_images[0].transpose(1, 2, 0).shape)
    cv2.imwrite("idata_test.png",training_images[29].transpose(1, 2, 0))
    cv2.imwrite("idata_test2.png",traj.images[0])
    print("img made")
    '''
