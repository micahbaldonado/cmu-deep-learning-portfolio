# %%
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import json
import torchvision.models.detection as detection
import torch
import time


coco_root = "./coco"
val_im_path = f"{coco_root}/val2017/val2017"
val_annotations_path = f"{coco_root}/annotations_trainval2017/annotations/instances_val2017.json"

with open(val_annotations_path, 'r') as f:
    all_coco_data = json.load(f)

print(all_coco_data.keys())

all_val_images = all_coco_data['images']
all_val_annotations = all_coco_data['annotations']
all_val_categories = all_coco_data['categories']

class CocoDataset(Dataset):
    def __init__(self, data_folder, transform = None):
        self.data_folder = data_folder
        self.filenames = os.listdir(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_folder, self.filenames[idx])
        image = Image.open(image_path).convert("RGB")

        if (self.transform == True):
            image = self.transform(image)
            
        return image_path, image

# print(all_val_images[0])

transpose = transforms.Compose([
        transforms.ToTensor()
    ])

CocoVal_Images = CocoDataset(val_im_path, transform = transpose)
CocoVal_Dataloader = DataLoader(CocoVal_Images, batch_size=64, shuffle = False, num_workers = 2)

YOLO_model = YOLO("yolov8n.pt")  # load the pretrained value, since we are just concerned about mAP 

#########################

F_RCNN_model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
F_RCNN_model.eval() # we don't want the computation graph to be updated

# %%
# Preprocessing: get the coco annotations ready so that I can actually compare them to the yolo output.

# we need to link image id to filename and category id to classname.
image_id_to_filename = {img["id"]: img["file_name"] for img in all_val_images}
# category_id_to_classname = {cat["id"]: cat["name"] for cat in all_val_categories}


# there is a mismatch, between the how the class ids are managed by the ground truth annotations
# and the yolo model. Below, I adjust for this by linking the category ID to the classname.
category_id_to_classname = {cat["id"]: cat["name"] for cat in all_val_categories}
# category_id_to_classname = {i: cat["name"] for i, cat in enumerate(all_val_categories)}
coco_category_id_to_yolo = {cat["id"]: i for i, cat in enumerate(all_val_categories)} # To finish the pipeline, we also need to go fromm coco cateogry to yolo cat.
# we also need a way to go from the YOLO index to the coco category ID.
yolo_to_coco_category = {v: k for k, v in coco_category_id_to_yolo.items()}


print("YOLO to COCO classes:", coco_category_id_to_yolo)



# Ultimately, we want to link filenames to all the annotation, gt data.
image_annotations = {} # dictionary with format {image_id : gt_annotations}

for annotation in all_val_annotations:
    image_id = annotation["image_id"]
    image_filename = image_id_to_filename[image_id] # we want to link the filenames with the annotations.
    class_id = annotation["category_id"]
    class_name = category_id_to_classname[class_id]
    # print(image_id)

    
    bb_coco = annotation["bbox"]
    x_min, y_min, bb_width, bb_height = bb_coco  
    # Since YOLO writes bounding boxes as [xmin, ymin, xmax, ymax], we need to convert from [xmin, ymin, width, height]
    x_max = x_min + bb_width
    y_max = y_min + bb_height

    # we don't want to include these. based on my research, you can
    # compute mAP without worrying about them.
    is_crowd_annotation = annotation["iscrowd"]
    if is_crowd_annotation: # crowd annotations can be skipped for the mAP metric
        continue

    gt_annotations = {
        "class_id": class_id, 
        "class_name": class_name, 
        "bb": [x_min, y_min, x_max, y_max]  # YOLO format
    }

     # initialize the dictionary key as an empty list so that image filenames can be linked to the ground truth annotations.
    if image_filename not in image_annotations:
        image_annotations[image_filename] = [] 

    image_annotations[image_filename].append(gt_annotations)

# sanity check: are these actually liking the filename to the annotaiton? -> YES
for i, (filename, annotations) in enumerate(image_annotations.items()):
    print(f"{filename}: {annotations}")
    if i == 4:
        break

class_names = [category['name'] for category in all_val_categories]
all_APs = { # We will store the preciaion and recall as a tuple for each element
    cls:[] for cls in class_names
} # now, APs is a dictionary, which can dynamically carry AP values to average over.
# the idea is to link one array that can be extended dynamically per class
# from here, even if not all classes show up in a single detection image,
# you can still extend the array AP values which do have that class present in the image.

# %% [markdown]
# ## Starting with Latency

# %%
# CocoVal_LATENCY_Dataloader = DataLoader(CocoVal_Images, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
# we need this to have a batch size of 1

LATENCY_CocoVal_Images = CocoDataset(val_im_path, transform = transpose) 
# this implicitly should work

latency4YOLO = []
latency4FasterRCNN = []

counter = 0
with torch.no_grad():  
    for _, (path, image) in enumerate(LATENCY_CocoVal_Images):
        
        # print(type(image))
        # print(len(image))
        # print(image)

        # we need the PIL image in the form of a tensor
        # image = transpose(image)
        # yolo_im = image.unsqueeze(0) # F-RCNN doesn't need this bc it exects images in a list

        counter += 1
                
        YOLO_start = time.time()
        YOLO_output  = YOLO_model(image) # YOLO is able to handle a PIL image and convert it to a tensor
        # we want to subtract how much time it has taken for YOLO to predict from a single image.
        YOLO_latency = time.time() - YOLO_start
        latency4YOLO.append(YOLO_latency)

        # However, for FRCNN, you need to convert from PIL Image to Tensor!!
        image = transpose(image) 

        FRCNN_start = time.time()
        # F_RCNN expects a lists of tensors, but we just need to put image in a list.
        FRCNN_output  = F_RCNN_model([image])
        # we want to subtract how much time it has taken for YOLO to predict from a single image.
        FRCNN_latency = time.time() - FRCNN_start
        latency4FasterRCNN.append(FRCNN_latency)

        if(counter > 100):
            break
        
        if(counter % 10 == 0):
            print(f"iteration #{counter}")
            print(f"Cur YOLO latency: {YOLO_latency}")
            print(f"F-RCNN latency: {FRCNN_latency}")
            
YOLOlatency_avg = np.average(latency4YOLO)
FasterRCNNlatency_avg = np.average(latency4FasterRCNN)


# %%
# LATENCY RESULTS:

print('Across 100 samples,')
print(f"Average YOLO Latency: {YOLOlatency_avg:.3f} seconds!")
print(f"Average Faster_rcnn Latency: {FasterRCNNlatency_avg:.3f} seconds!")

# %% [markdown]
# ## mAP Scores

# %% [markdown]
# ### Need a way to calculate intersection of union

# %%
def IoU(bb1, bb2):
    x_min1, y_min1, x_max1, y_max1 = bb1
    x_min2, y_min2, x_max2, y_max2 = bb2
    
    x_span1 = x_max1-x_min1
    y_span1 = y_max1-y_min1
    b1_area = x_span1 * y_span1

    x_span2 = x_max2-x_min2
    y_span2 = y_max2-y_min2
    b2_area = x_span2 * y_span2

    x_shared_span = max(0, min(x_max1, x_max2) - max(x_min1, x_min2))
    y_shared_span = max(0, min(y_max1, y_max2) - max(y_min1, y_min2))

    small_intersect = x_shared_span * y_shared_span
    union = b1_area + b2_area - small_intersect

    return small_intersect/union

# test case:
bb1 = [90, 50, 150, 150] 
bb2 = [80, 100, 100, 200] 
print("IoU:", IoU(bb1, bb2))


# %%
for IoU_thesh in np.arange(0.5, 1, 0.05): # no box will ever be perfect so we don't include 1
    for conf_thresh in np.arange(0, 1.05, 0.05): # we don't want all ranges of cconfidence
        # for (paths, images) in enumerate(CocoVal_Dataloader): 
        # for (path, image) in zip(paths, images): # rather than ennumerate, this is how we can go through each path
        
        ###################################################
        N = 1 # since I need to run this on all images, this number needs to be repolaced
        # by the number of images in the COCO Val dataset. The only reason I am using 1 is so I can
        # check my work as I work on this problem.
        ###################################################

        for i in range(N): 
            path, image = CocoVal_Images[i]
            # and image within each batch.

            print("Sample keys in image_annotations:", list(image_annotations.keys())[:10])


            results = YOLO_model(image)

            #converting the tensors to numpy since gt is numpy
            boxes_info = results[0].boxes
            bounding_boxes = boxes_info.xyxy.numpy()
            confidence_scores = boxes_info.conf.numpy()
            classes = boxes_info.cls.numpy()

            # putting all these variables together, so I can sort them and keep track of everything.
            YOLO_predictions = list(zip(confidence_scores, bounding_boxes, classes))
            # we need to sort the list based on confidence scores
            # we need to make sure the confidence scores decrease.
            YOLO_predictions.sort(key=lambda x: x[0], reverse=True)


            # getting gt data: to get the path in the same format, I need to make sure that the zeros 
            # are stripped and that the file is 12 characters long before the jpg
            cur_id = path.split("\\")[-1].split(".")[0]  
            twelve_digit_id = f"{int(cur_id):012d}" 
            image_name = f"{twelve_digit_id}.jpg" # now it's in the same format as gt annotaitons!

            gt = image_annotations[image_name]
            
            gt_bbs = [g['bb'] for g in gt]
            # gt_classes = [g['class_name'] for g in gt] 
            # need a way to convert to the actual class. Make use of earlier function.
            gt_classes = [coco_category_id_to_yolo[g['class_id']] for g in gt]


            TP = 0
            FP = 0

            best_match_indeces = set()
            # we only want to match a single prediciton.
            # everything else is irrelevant, so we use a set instead of a list.

            for conf, bb_pred, cur_class in YOLO_predictions:
                # cur_class = int(cur_class)
                # cur_class = yolo_to_coco_category[int(cur_class)]
                cur_class_coco = yolo_to_coco_category[int(cur_class)]


                # now we can go from yolo to coco cateogry!

            
                if(conf < conf_thresh):
                    continue

                # we want to find the best GT match with our predictions:
                best_IoU = 0
                ideal_match_idx = -1
                for (match_idx, gt_bb) in enumerate(gt_bbs):
                    cur_IoU = IoU(bb_pred, gt_bb)
                    if(cur_IoU > best_IoU):
                        best_IoU = cur_IoU
                        ideal_match_idx = match_idx
                
                if(ideal_match_idx != -1 and best_IoU > IoU_thesh
                   and cur_class == gt_classes[ideal_match_idx] # we need to make sure the class matches before we consider this a true positive.
                   ):
                    TP += 1 # it is a true positive if it matches perfectly
                    best_match_indeces.add(ideal_match_idx) # you use add with set, not append.
                else:
                    FP += 1

            FN = len(gt_bbs) - len(best_match_indeces)

            # to avoid divison by 0:

            if(TP + FP > 0):
                cur_precision = TP/(TP+FP)
            else:
                cur_precision = None

            if(TP + FN > 0):
                cur_recall = TP/(TP+FN)
            else:
                cur_recall = None

            if cur_precision is not None and cur_recall is not None:
                # uWe want to use the correct cooo class name.
                # cur_class_name = category_id_to_classname[cur_class] 
                cur_class_name = category_id_to_classname[cur_class_coco] 


                # we need to populate all_APs if the class is not already present.
                if cur_class_name  not in all_APs:
                    all_APs[cur_class_name] = []
                
            # store precision and recall as a tuple
            PR_tuple = (cur_precision, cur_recall)
            all_APs[cur_class_name].append(PR_tuple)


# %% [markdown]
# ### to compute mAP from all_APs, we need to use trapezoidal integration

# %%
# You can use numpy to perform trapezoidal integration

import numpy as np

def get_actual_AP(precision, recall):
    if(len(precision) > 1):
        return np.trapz(precision, recall) 
    else:
        return 0


AP_before_mean = {}

for class_name, precision_recall_vals in all_APs.items():
    existent_pr_values = [] # we want to fill this with PR values for each class.

    for PR_tuple in precision_recall_vals:
        if PR_tuple[0] is not None and PR_tuple[1] is not None:  
            # I was runnning into an issue where the precision
            # and recall values kept returning as None.
            # This way, I can skip these values as they
            # are not supposed to contriubte to the AP curve.
            existent_pr_values.append(PR_tuple)

    if existent_pr_values:
        # we want to retrieve the precision, recall tuple values.
        precision, recall = zip(*existent_pr_values)
        AP_before_mean[class_name] = get_actual_AP(precision, recall)
    else:
        AP_before_mean[class_name] = 0.0 

print(AP_before_mean)

# print(all_APs)


# %% [markdown]
# ### Calcualte mAP for YOLO

# %%
mAP = sum(AP_before_mean.values())/len(AP_before_mean)

# %% [markdown]
# ### Calculate mAP for Faster-RCNN

# %%
# mAP_F_RCNN = sum(AP_before_mean.values())/len(AP_before_mean)


