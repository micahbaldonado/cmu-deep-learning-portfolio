import glob
import os

import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch


class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def get_data_loader(data_path, opts):
    """Create training and test data loaders."""
    basic_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # debug
    print(f"preprocessing type: {opts.data_preprocess}")

    if opts.data_preprocess == 'basic':
        train_transform = basic_transform
    elif opts.data_preprocess == 'advanced':

        # I used similar code for my data augmentation in hw3, but I modified it by
        # commenting out the lines of code which may confuse the generator.
        

        # NORMAL
        #################################################
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]

        train_transform = transforms.Compose([
        transforms.Resize(osize, Image.BICUBIC),
        # transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)), # my discriminator is overfitting with extremely low loss,
        # so I am adding this to confuse it more.
        # Basically, you can introduce gaussian noise to make the discriminator less confident, allowing the generator more room to
        # expirement with diversity
        transforms.RandomCrop(opts.image_size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding = 4), # we want to keep the size the same, but we want to obscure part of the image.
        # transforms.RandomRotation((-4, 4)),
        transforms.ColorJitter(brightness = 0.03, contrast = 0.05, saturation=0.07, hue = 0.03),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ##########################################################

        # WGAN
        # load_size = int(1.1 * opts.image_size)
        # osize = [load_size, load_size]

        # train_transform = transforms.Compose([
        # transforms.Resize(osize, Image.BICUBIC),
        # # transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)), # my discriminator is overfitting with extremely low loss,
        # # so I am adding this to confuse it more.
        # # Basically, you can introduce gaussian noise to make the discriminator less confident, allowing the generator more room to
        # # expirement with diversity
        # transforms.RandomCrop(opts.image_size),
        # transforms.RandomHorizontalFlip(),
        # # transforms.RandomCrop(32, padding = 4), # we want to keep the size the same, but we want to obscure part of the image.
        # # transforms.RandomRotation((-4, 4)),
        # # transforms.ColorJitter(brightness = 0.03, contrast = 0.05, saturation=0.07, hue = 0.03),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))




        # before, removing the noramlization step worked, but it created washed out images.
        # my new goal here is to prevent the images from washing out while preventing
        # mode collapse. I plan to achieve this adding a normalization factor but instead of 
        # restricting the input to [0, 1] or [-1, 1], I plan on restricting it to something inbetween.
        # transforms.Normalize((0.4, 0.4, 0.4), (0.3, 0.3, 0.3))

        # you can just look these values up. Normalization is an important preprocessing step
        
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        
        # we don't want to remove too much, but we want to make the learning process
        # more robust by introducing occlusions.
        ])


    dataset = CustomDataSet(
        os.path.join('data/', data_path), opts.ext, train_transform
    )
    dloader = DataLoader(
        dataset=dataset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers
    )

    return dloader
