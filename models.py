import torch
import torchrl.modules
import torchvision.transforms as transforms 
import torch.nn as nn
import numpy as np
from torchsummary import summary
import cv2

class Encoder(nn.Module):
    #variable based on input size, for this assessment we assume input 64x64x1
    def __init__(self):
        super(Encoder, self).__init__()
        # convolution layer w/stride of 2 to reduce dim, then ReLU activation. 
        # Can be changed later depending on performance
        #TODO: handle 3,64,64???
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride = 2, padding=1), #32
            nn.Tanh(),
            nn.Conv2d(4, 8, 3, stride = 2, padding=1), #16
            nn.Tanh(),
            nn.Conv2d(8, 16, 3, stride = 2, padding=1), #8
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride = 2, padding=1), #4
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, stride = 2, padding=1), #2
            nn.Tanh(),
            nn.Conv2d(64, 128, 3, stride = 2, padding=1), #1
            nn.Tanh(),
            nn.Flatten()
        )
        self.linear = nn.Linear(128, 64)
        #TODO: ^ write this as a append loop
        # TODO: fix breaking on batch size of 2 (torchsummary default)
        #TODO: make this work without transpose

    def forward(self, image):
        # print(type(image))
        image = self.cnn(image)
        # image = nn.Flatten(image)
        # print(image.shape)
        image = self.linear(image)
        # set back to channels
        # print(image.shape)
        return image

class Decoder(nn.Module):
    #detached per paper specs
    #reverse Encoder architecture
    def __init__(self):
        super(Decoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 4, 3, stride=2),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 2, 3, stride=2),
            nn.BatchNorm2d(2),
            nn.ConvTranspose2d(2, 1, 3, stride=2),
            nn.BatchNorm2d(1),
            nn.Conv2d(1,1,3,stride=2, padding=1),
            nn.Tanh()
        )
        self.linear = nn.Linear(64,128)
    
    def forward(self, embedding):
        # TODO: add linear??
        return self.cnn(embedding)

class RewardModel(nn.Module):
    #links together encoder, MLP, feeds forward. etc.
    def __init__(self, parameters): # params is a dict of parameters
        super(RewardModel, self).__init__()
        #TODO: device binding for when training, add .toDevice as well as self.device
        # set all hyperparameters from paper
        self.timesteps = parameters['t']
        self.k = parameters['k']
        self.oc_mlp = parameters["out_channels_mlp"]
        self.is_lstm = parameters["input_size_lstm"]
        #assign encoder and mlp to respective classes
        self.encoder = Encoder()
        self.a_mlp = nn.Tanh()
        #in features is 64, outputs 64, 3 linear layers, defaults to tanh activation
        self.mlp = torchrl.modules.MLP(64,64,2)
        #TODO: make sure u actually understand what LSTM is doing here
        # take first MLP output (from all encoders) and pass to LSTM
        self.lstm = nn.LSTM(input_size=self.is_lstm, hidden_size=self.oc_mlp, num_layers=1, batch_first=True)

    def forward(self,img,bs=2): #img input = batch size of time steps of 1x64x64
        print("fwd")

if __name__ == '__main__':
    # check shape, sequence
    encoder = Encoder()
    # summary(encoder,(3,64,64),batch_size=7)
    decoder = Decoder()
    summary(decoder,(64,1,1),batch_size=30)
    '''
    img = cv2.imread("idata_test2.png", cv2.IMREAD_GRAYSCALE)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img)
    encoder_gen = encoder(tensor)
    print(encoder_gen.shape)
    decoder = Decoder()
    summary(decoder, encoder_gen.shape)
    '''
