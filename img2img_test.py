from models import *
from dataloading import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import os

#testing encoder and decoder arch to make sure they actually work
#device binding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#create dataset
spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=4,      # number of shapes per trajectory
        rewards=[ZeroReward],
    )

dataset = dataloader(spec,bs=40,len=80)
test_set = dataloader(spec,bs=20,len=1)
lr = 0.0001

'''
print('training')
for img_batch, rewards in dataset:
    print(img_batch.shape)
    for img_seq in img_batch:
        print(img_seq.shape)
        display = torchvision.utils.make_grid(img_seq)
        display = torchvision.transforms.ToPILImage()(display)
        display.show()
print('test')
for img_batch, rewards, in test_set:
    print(img_batch.shape)
'''

encoder = Encoder().to(device)
decoder = Decoder().to(device)

criterion = nn.MSELoss()
# print(list(encoder.parameters())) model tensors!
# print(list(decoder.parameters())) who would've guessed
autoencoder = nn.Sequential(encoder,decoder)
optimizer = optim.AdamW(autoencoder.parameters(),lr=lr)
# ^ optimizer for both, default to Adam but could use other

epochs = 64

best_val_loss = float("inf")

for epoch in range(epochs):
    print(f'Epoch: {epoch} / {epochs}')
    # set to training mode
    encoder.train()
    decoder.train()

    for batch,rewards in dataset:
        running_loss = 0.0 # set loss for this epoch
        for data in batch:
            data = transforms.Grayscale(num_output_channels=1)(data)
            # print(data.shape)
            data = data.to(device) # pass batch to device
            # print(f'data max/min {data.max()},{data.min()}')
            encoded = encoder(data) #feed forward
            # print(f'encoded max/min {encoded.max()},{encoded.min()}')
            # print(encoded.shape)
            encoded = encoded[:,:,None,None]
            # print(encoded.shape)
            decoded = decoder(encoded)
            #print(decoded.shape)
            # print(f'decoded max/min {decoded.max()},{decoded.min()}')

            loss = criterion(decoded, data)
            # print(loss.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss/len(batch)
        print(f'avg training loss this epoch: {train_loss}')

        if train_loss < best_val_loss:
            best_val_loss = train_loss
        torch.save(
            {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
            "model_weights/autoencoder.pt",
        )

encoder.eval()
decoder.eval()

for img_batch, rewards in test_set:
    for img_seq in img_batch:
        img_seq = transforms.Grayscale(num_output_channels=1)(img_seq)
        original = img_seq
        encoded = encoder(img_seq)[:,:,None,None]
        img_seq = decoder(encoded)
        display = list(img_seq) + list(original)
        display = torchvision.utils.make_grid(display,nrow=30)
        display = torchvision.transforms.ToPILImage()(display)
        display.show()