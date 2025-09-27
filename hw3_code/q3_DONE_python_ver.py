import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import cv2

# CANNOT USE CV2; must bound the boxes

# dog is kopek in turkish
# cat is kedi in turksih
# dog is 2007_001239.jpg
# cat is 2008_002152.jpg



kopek_path = '2007_001239.jpg'
kedi_path = '2008_002152.jpg'

kopek_image = Image.open(kopek_path) # keep image in RGB, as resnet expects 3 channels.
kedi_image = Image.open(kedi_path)
kopek_np = np.array(kopek_image) # shape: (375, 500)?
kedi_np = np.array(kedi_image) # shape: (375, 500)?

print('kopek shape:', np.shape(kopek_np)) 

# This should be the easiest window sliding method:
# DOG = KOPEK
kopek_window_size = (120, 120, 3)
kopek_window_h = kopek_window_size[0]
kopek_window_w = kopek_window_size[1]
kopek_step_size = 70 # it's less expensive to slide every X amount of pixels. 


# CAT = KEDI
kedi_window_size = (140, 140, 3) # (150, 150, 3) works with tiger cat
kedi_window_h = kedi_window_size[0]
kedi_window_w = kedi_window_size[1]
kedi_step_size = 60 # 60 is decent for tabby # 70 is decent for tiger cat

kopek_windows_np = np.lib.stride_tricks.sliding_window_view(kopek_np, kopek_window_size)[::kopek_step_size, ::kopek_step_size]
kedi_windows_np = np.lib.stride_tricks.sliding_window_view(kedi_np, kedi_window_size)[::kedi_step_size, ::kedi_step_size]
print("\nkopek window shape:", kopek_windows_np.shape)
print("\nkedi window shape:", kedi_windows_np.shape)


# Loading the model: using Resnet because it's strong for classification but also not overkill
weights = ResNet18_Weights.DEFAULT
meta_transforms = weights.transforms() # use the meta-knowledge on how to transform the input images. 
model = resnet18(weights = weights)
model.eval() # print(model)

preprocess  = transforms.Compose([
    transforms.Resize((224, 224)), # dinputs of resnet18 are in this size.
    transforms.ToTensor(),
    transforms.Normalize(mean=meta_transforms.mean, std=meta_transforms.std)  # Use metadata values
])

imagenet_classes = weights.meta["categories"]
dog_idx = imagenet_classes.index("redbone")
# cat_idx = imagenet_classes.index("tiger cat")
cat_idx = imagenet_classes.index("tabby")

print(dog_idx)
print(cat_idx)

all_preds = []
all_probs = []

# Image.fromarray expects (H, W, 3)
kopek_flattened_windows = kopek_windows_np.reshape(-1, kopek_window_h, kopek_window_w, 3)
kedi_flattened_windows = kedi_windows_np.reshape(-1, kedi_window_h, kedi_window_w, 3)



kopek_windows_tensor = torch.tensor(kopek_flattened_windows, dtype=torch.float32) # / 255.0  
kopek_windows_tensor = kopek_windows_tensor.permute(0, 3, 1, 2)

batch_size = 64
kopek_num_batches = len(kopek_flattened_windows) // batch_size + 1
print("NUM BATCHES", kopek_num_batches)

dog_best_indeces = []
kopek_threshold = 0.90

for i in range(kopek_num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(kopek_flattened_windows)) # we want to include all images (we use minimum to ensure this.)
    batch = kopek_flattened_windows[start:end]

    batch_tensor = torch.tensor(batch, dtype=torch.float32) 
    batch_tensor = batch_tensor.permute(0, 3, 1, 2)
    batch_tensor = torch.stack([preprocess(Image.fromarray(img.astype('uint8'))) for img in batch])

    with torch.no_grad():
        output = model(batch_tensor)

    # print("Output size", output.size()) # [64 x 1000]
    probs = F.softmax(output, dim=1)
    dog_probs = probs[:, dog_idx]
    # print(dog_probs)

    indeces = (dog_probs > kopek_threshold).nonzero().squeeze().numpy()
    # print(indeces)

    # print(dog_probs[dog_probs > threshold])
    # indeces = 
    indeces = np.atleast_1d(indeces) # ensure that indeces is an array. this way extending is easier.
    dog_best_indeces.extend(start + indeces)

print(dog_best_indeces)


dog_guess = kopek_flattened_windows[dog_best_indeces[0]]



# # Given an index from one of the best performing indeces in the classifier,
# # this function produces the bounding box coordinates consistent with the window
# # function. The main idea is that you can backtrack which window slice is relevant from the index, and
# # all you need is the index to figure out the important. coordinates.
def get_bounding_box(index, window_h, window_w, step_size, num_cols):
    row = index // num_cols
    col = index % num_cols

    x1 = col * step_size
    y1 = row * step_size
    x2 = x1 + window_w
    y2 = y1 + window_h

    return (x1, y1, x2, y2)

# index = dog_best_indeces[0]


# Drawing the boxes for kopek (dog)
image = Image.open(kopek_path)

fig, ax = plt.subplots()
ax.imshow(image)

for i in dog_best_indeces:
    num_cols = (image.size[0] - kopek_window_w) // kopek_step_size + 1
    bounding_box = get_bounding_box(i, kopek_window_h, kopek_window_w, kopek_step_size, num_cols)
    bb = tuple(map(int, bounding_box))
    rectangle = patches.Rectangle((bb[0], bb[1]), kopek_window_w, kopek_window_h,
                             linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rectangle)

    ax.text(bb[0], bb[1] - 5, "Redbone", fontsize=8, color='white', 
            bbox=dict(facecolor='green', alpha=0.5, edgecolor='none', boxstyle="round,pad=0.3"))

plt.show()

# ########################################################################################

# CAT IMAGE:

kedi_windows_tensor = torch.tensor(kedi_flattened_windows, dtype=torch.float32) # / 255.0  
kedi_windows_tensor = kedi_windows_tensor.permute(0, 3, 1, 2)

batch_size = 64
kedi_num_batches = len(kedi_flattened_windows) // batch_size + 1
print("NUM BATCHES", kedi_num_batches)

cat_best_indeces = []
kedi_threshold = 0.007 # 0.03 was decent for tiger
# Resnet18 is really bad at identifying cats. 
# Even if you try a whole image classification, the top predictions are the following:

# Top 5 Predictions:
# tiger cat: 0.2260
# wood rabbit: 0.1280
# tabby: 0.1035
# wallaby: 0.0562
# French bulldog: 0.0470

# Therefore, you must use extremely low probabilities.
# However, the signal seems to pick up fur at extremely low probabiltiies
# which makes sense, since it's an important feature.

for i in range(kedi_num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(kedi_flattened_windows)) # we want to include all images (we use minimum to ensure this.)
    batch = kedi_flattened_windows[start:end]

    batch_tensor = torch.tensor(batch, dtype=torch.float32) 
    batch_tensor = batch_tensor.permute(0, 3, 1, 2)
    batch_tensor = torch.stack([preprocess(Image.fromarray(img.astype('uint8'))) for img in batch])

    with torch.no_grad():
        output = model(batch_tensor)

    probs = F.softmax(output, dim=1)
    cat_probs = probs[:, cat_idx]
    print(cat_probs)


    indeces = (cat_probs > kedi_threshold).nonzero().squeeze().numpy()

    indeces = np.atleast_1d(indeces) # ensure that indeces is an array. this way extending is easier.
    cat_best_indeces.extend(start + indeces)

print(cat_best_indeces)


image = Image.open(kedi_path)

fig, ax = plt.subplots()
ax.imshow(image)

for i in cat_best_indeces:
    num_cols = (image.size[0] - kedi_window_w) // kedi_step_size + 1
    bounding_box = get_bounding_box(i, kedi_window_h, kedi_window_w, kedi_step_size, num_cols)
    bb = tuple(map(int, bounding_box))
    rectangle = patches.Rectangle((bb[0], bb[1]), kedi_window_w, kedi_window_h,
                             linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rectangle)

    ax.text(bb[0], bb[1] - 5, "Tabby", fontsize=8, color='white', 
            bbox=dict(facecolor='green', alpha=0.5, edgecolor='none', boxstyle="round,pad=0.3"))

plt.show()