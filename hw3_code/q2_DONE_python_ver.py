# %% [markdown]
# # Start Code

# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.resnet import BasicBlock

# %%
# model = resnet18(pretrained = False)
# print(model)


# After some research, I realized that while resnet is awesome for large
# images, it's actually kind of clunky and aggressive. For example, the first
# layer of resnet starts with a kernel size of (7x7) and it's whole architecture
# invovles aggressive downsampling since it's trained on insanely large images
# compared to CIFIR 100. With this in mind, I am gonna do something much more 
# practical. Since we can import pretrained models without any penalty, this also
# means we caan import the residual blocks from pretrained models without any penalty
# I am going ot make use of thi by basically writing my own deep network but personally
# tailored for smaller images. I'm going to be very kind and gentle to these little
# images unlike the aggressive resnet18.

# I want to use BasicBlock since it already handles the addition (not concatenation) of
# of the skip connections, which is what I plan to leverage to get above 60%


class MyResnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_hidden, k):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_num_neuron = num_hidden
    
        # k = 150 # arbitrary

        # reduce from 32 x 32 to 16 x 16
        # self.conv1 = nn.Conv2d(in_channels, k, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv1 = nn.Conv2d(in_channels, k, kernel_size=3, stride=1, padding=1, bias=False) 

        self.batch_norm = nn.BatchNorm2d(k) # adding this to prevent vanishing gradients.
        self.act1 = nn.ReLU()
        # (16 x 16) now to (8 x 8):
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)

        # REMOVED AND REPLACED, SAME WITH FIRSTS CONV
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  

       
        # I ran into some problems; i would need to do the convolution of kernel 1 for the concatenation
        # rather than do that, I'll just start with a basic block
        # self.basic_block_0 = BasicBlock(in_channels, k) 

        # every basic block has 2 3
        self.residual_blocks_1 = nn.Sequential(
            # we need to modify the basic block since we're gpoing from k to 32 connections. This requires downsampling.
            BasicBlock(k, k * 2, downsample= nn.Conv2d(k, k * 2, kernel_size=1, stride=1)),
            BasicBlock(k * 2, k * 2)
        )

        # now I can downsample a little.
        # reduce from (8 x 8) to (4 x 4)
        self.residual_blocks_2 = nn.Sequential(
            # on a very high level, all that I need to do is whenever I change the input to output channels,
            # I can straight up just downsample from the same output to input
            # dimension, same stride, while making sure that the kernel size is 1
            # to enable the matrix sizes for the skip connection and the input matrix to be correct.
            BasicBlock(k * 2, k * 3, stride = 2, downsample= nn.Conv2d(k * 2, k * 3, kernel_size=1, stride=2)),
            BasicBlock(k * 3, k * 3)
        )

        self.residual_blocks_3 = nn.Sequential(
            BasicBlock(k * 3, k * 3, stride = 2, downsample= nn.Conv2d(k * 3, k * 3, kernel_size=1, stride=2)),
            BasicBlock(k * 3, k * 3)
        )

        # (4 x 4) to (2 x 2)
        self.residual_blocks_4 = nn.Sequential(
            BasicBlock(k * 3, k * 4, stride = 2, downsample= nn.Conv2d(k * 3, k * 4, kernel_size=1, stride=2)),
            BasicBlock(k * 4, k * 4)
        )

        # 
        # self.residual_blocks_5 = nn.Sequential(
        #     BasicBlock(128, 256, stride = 2, downsample= nn.Conv2d(128, 256, kernel_size=1, stride=2)),
        #     BasicBlock(256, 256),
        # )

        # we need to reduce the channel depth to 1 in order to
        # finish things off with a FC layer to get the class
        # probabilities.

        # this finally brings from (2x2) to (1x1)
        self.AveragePooling = nn.AdaptiveAvgPool2d(1)  # rather than manually put the kernel size, you can jsut autaotmically scale the input to 1

        with torch.no_grad():
            dummy_tensor = torch.randn(1, in_channels, 32, 32) # just minmic input dimensions
            x = dummy_tensor
            x = self.conv1(x)
            x = self.batch_norm(x)
            x = self.act1(x)
            x = self.maxpool(x)
            # x = self.basic_block_0(x)
            x = self.residual_blocks_1(x)
            x = self.residual_blocks_2(x)
            x = self.residual_blocks_3(x)
            x = self.residual_blocks_4(x)
            # x = self.residual_blocks_5(x)
            x = self.AveragePooling(x)
            x = torch.flatten(x, 1) # flatten from (batch_size, num_channel, 1, 1) to (batch, num_filters/channesl)
            x_size = x.size(1) # this grabs the feature size for the final linear layer.

        # i don't know what the shape of linear is, but I'll just use a dummy tensor to figure that out
        self.linear1 = nn.Linear(x_size, num_hidden // 2) # we just need to know the kernel size. I calculated this manually.
        # we need to end up wtih a channel depth of 1, so this has to end with shape (1, 1) from the prev block.
        # cross entropy loss can take it form here.

        self.dropout = nn.Dropout(0.3) # going for dropout since my previous cases were overfitting a ton
        # self.linear2 = nn.Linear(num_hidden, num_hidden // 2)
        self.linear3 = nn.Linear(num_hidden // 2, out_channels)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.act1(x)
        x = self.maxpool(x)
        # x = self.basic_block_0(x)
        x = self.residual_blocks_1(x)
        x = self.residual_blocks_2(x)
        x = self.residual_blocks_3(x)
        x = self.residual_blocks_4(x)
        # x = self.residual_blocks_5(x)
        x = self.AveragePooling(x)
        x = torch.flatten(x, 1) # flatten from (batch_size, num_channel, 1, 1) to (batch, num_filters/channesl)
        x = self.linear1(x)
        x = self.dropout(x)
        # x = self.linear2(x)
        x = self.linear3(x)

        return x

# %% [markdown]
# # Load the data

# %%
augmentation = transforms.Compose([
    transforms.RandomCrop(32, padding = 4), # we want to keep the size the same, but we want to obscure part of the image.
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation((-4, 4)),
    transforms.ColorJitter(brightness = 0.05, contrast = 0.05, saturation=0.07, hue = 0.06),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    # you can just look these values up. Normalization is an important preprocessing step
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    # we don't want to remove too much, but we want to make the learning process
    # more robust by introducing occlusions.
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # you can just look these values up. Normalization is an important preprocessing step
])


train_dataset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform=augmentation)
test_dataset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True, num_workers = 2) # num_workers is used to speed up training via preloading
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False, num_workers = 2)

sample_image, sample_label = train_dataset[0]
print(sample_image.shape)

# %%
# TRY LARGER K
model = MyResnet(in_channels=3, out_channels=100, num_hidden = 100, k = 32)
loss_fc = torch.nn.CrossEntropyLoss() # by default, this should aggregate all the loss for the entire batch.

epochs = 70 # doesn't need to be insanely high, since the model sees so many images every epoch.

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=2e-3, nesterov=True)
# sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001)
# optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.05)
# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
# sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
# sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
# for many epochs, cosine annearling with warm restarts is strong. by having a very dynamic
# lr, it can get out of local minima. the warm restarts are to prevent the lr changes from
# being too dynamic.

# SGD and CSA_lr is apparently a good combo

train_accuracy = []
test_accuracy = []
train_loss  = []
test_loss = []

best_test_acc = 0

for i in range(epochs):
    cur_train_loss = []
    cur_train_accuracy = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        model.train()
        # print(batch_idx)

        model_pred = model(images)
        loss = loss_fc(model_pred, labels) # predictions first for loss_fc
        cur_train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        FCNN_accuracy = (torch.argmax(model_pred, dim=1) == labels).float().mean().item()
        cur_train_accuracy.append(FCNN_accuracy)
        
    cur_test_loss = []
    cur_test_accuracy = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            model.eval()

            model_pred = model(images)
            loss = loss_fc(model_pred, labels) 
            cur_test_loss.append(loss.item())

            FCNN_accuracy = (torch.argmax(model_pred, dim=1) == labels).float().mean().item()
            cur_test_accuracy.append(FCNN_accuracy)

    # aggregate metrics

    train_avg_acc = np.average(cur_train_accuracy)
    train_accuracy.append(train_avg_acc)
    test_avg_acc = np.average(cur_test_accuracy)
    test_accuracy.append(test_avg_acc)

    train_avg_loss = np.average(cur_train_loss)
    train_loss.append(train_avg_loss)
    test_avg_loss = np.average(cur_test_loss)
    test_loss.append(test_avg_loss)

    sched.step(test_avg_loss)

    if(test_avg_acc > best_test_acc):
        best_test_acc = test_avg_acc
        torch.save(model.state_dict(), "best_test_model.pth")

    print(f"Epoch {i+1} train loss for MyResnet: {train_avg_loss:.4f}")
    print(f"Epoch {i+1} train accuracy for MyResnet: {train_avg_acc:.4f}")
    print(f"Epoch {i+1} test loss for MyResnet: {test_avg_loss:.4f}")
    print(f"Epoch {i+1} test accuracy for MyResnet: {test_avg_acc:.4f}")

# %%
# Note: I stopped at the 46th epoch because I got these awesome results!:
# Epoch 46 train loss for MyResnet: 1.4050
# Epoch 46 train accuracy for MyResnet: 0.6138
# Epoch 46 test loss for MyResnet: 1.3892
# Epoch 46 test accuracy for MyResnet: 0.6130

model = MyResnet(in_channels=3, out_channels=100, num_hidden=100, k=32)
model.load_state_dict(torch.load("best_test_model.pth"))
model.eval()

# %%
# PLOTS of MYRESNET, MAX ACCURACY + MODEL ARCHITECTURE, AND HYPERPARAMETERS

best_epoch = 46

fig, ax = plt.subplots(1, 2, figsize=(12, 5))  
ax[0].plot(range(best_epoch), train_loss, color='tab:blue', label="Train Loss")
ax[0].set_title("Train Loss of My Resnet")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()
ax[1].plot(range(best_epoch), test_loss, color='tab:red', label="Test Loss")
ax[1].set_title("Test Loss of My Resnet")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()
plt.show()

max_train_acc = max(train_accuracy)
max_test_acc = max(test_accuracy)

print("Below is the Resnet model architecture")
print(model)

print("Relevant hyperparameters:")
print(f"Epochs (this was the best epoch, where I stopped training): {best_epoch}")
print(f"Optimizer: {type(optimizer).__name__}")
print(f"Loss Function: {loss_fc.__class__.__name__}")
print(f"Learning Rate: {0.005}")

# print(f"Activation Function: ReLU! I just wanted to go for simplicity")
      
print(f"The max training accuracy of my Resnet was {max_train_acc}.")
print(f"The max testing accuracy of my Resnet was {max_test_acc}.")

# %% [markdown]
# ## Visualization of Predications

# %%
images, labels = next(iter(train_loader))
first10_images_train = images[0:10]
first10labels_train = labels[0:10]

model.eval()
with torch.no_grad():
    FCNN_pred_idx = model(first10_images_train).argmax(dim=1) 

plt.figure(figsize=(4, 20))
for i in range(10):
    plt.subplot(10, 1, i + 1)
    plt.imshow(first10_images_train[i].permute(1, 2, 0))  # We need the images in form (H, W, C) - not (C, H, W)
    plt.title(f"Actual Class:{test_dataset.classes[first10labels_train[i]]}; Predicted Class: {test_dataset.classes[FCNN_pred_idx[i]]}")
    plt.axis("off")


plt.suptitle(f"Predictions from My Resnet on Train Data: { 100*(FCNN_pred_idx == first10labels_train).float().mean().item():.4f}% Accuracy!")
plt.show()

images, labels = next(iter(test_loader))
first10_images = images[10:20]
first10labels = labels[10:20]

model.eval()
with torch.no_grad():
    FCNN_pred_idx = model(first10_images).argmax(dim=1) 

plt.figure(figsize=(4, 20))
for i in range(10):
    plt.subplot(10, 1, i + 1)
    plt.imshow(first10_images[i].permute(1, 2, 0))  # We need the images in form (H, W, C) - not (C, H, W)
    plt.title(f"Actual Class:{test_dataset.classes[first10labels[i]]}; Predicted Class: {test_dataset.classes[FCNN_pred_idx[i]]}")
    plt.axis("off")

plt.suptitle(f"Predictions from My Resnet on Test Data: {100*(FCNN_pred_idx == first10labels).float().mean().item():.4f}% Accuracy!")
plt.show()


