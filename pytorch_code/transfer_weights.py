import torchvision.models as models
from dataset import Youtube8m
from models_src.weight_transfer import Weight_Transfer
from models_src.t3d import DenseNet3D

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

densenet_2D = models.densenet201(pretrained=True)
#should probably use higher resolution inputs since our actions take up a smaller space in the image
t3d = DenseNet3D(num_init_features=96, growth_rate=48, block_config=(6, 12, 48, 32), drop_rate=0.2, classifier=False)
transfer = Weight_Transfer(densenet_2D, t3d, twoD_out_features=1000, threeD_out_features=t3d.get_num_out_features(), frames_per_batch=60)
#print(transfer)

data = Youtube8m("data/ucf-arg/aerial_clips/boxing", vid_dim=224, frames_per_batch=60)
#data = Youtube8m()

## hyperparameters
epochs = 30
print_step = len(data) // 10
save_step = 5
device = torch.device('cuda:0')
transfer = transfer.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transfer.parameters(), lr=0.001)

for e in range(epochs):
    running_loss = 0.0
    total_train_loss = 0

    for i, frames in tqdm(enumerate(data, 0)):

        inputs, labels = frames
        print(inputs.shape)
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()

        out = transfer(inputs)
        loss_size = loss(out, labels)
        loss_size.backward()
        optimizer.step()

        running_loss += loss_size.data[0]
        total_train_loss += loss_size.data[0]

        if (i + 1) % (print_step + 1) == 0:
            print("Epoch: {}/{} \t train_loss: {:.2f}".format(e + 1, int(100 * (i+1) / len(data)), running_loss / print_every))

    if e % save_step == 0:
        print("Saving t3d model weights...")
        torch.save(t3d.state_dict, "saved_weights/t3d_transfer.pth")


print("training done")
