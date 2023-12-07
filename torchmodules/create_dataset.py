import os
import cv2
from torchvision.transforms import v2
import torch

from PIL import Image
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

import preprocessing

NUM_WORKERS = os.cpu_count()

class CustomTransform:
    """"
    https://www.python-engineer.com/courses/pytorchbeginner/10-dataset-transforms/
    """
    def __init__(self, IMG_SIZE, sigmaX):
        self.sigmaX = sigmaX
        self.IMG_SIZE = IMG_SIZE
        self.output_folder = "preprocessed"
    
    
    def __call__(self, img):
        # Preprocess with ben preprocessing
        img = np.array(img)
        img = self.load_ben_color(img, self.IMG_SIZE, self.sigmaX)

        return img


    def crop_image_from_gray(self, img, tol=7):
        if img.ndim ==2:
            mask = img>tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = gray_img>tol        
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0):
                return img
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img


    def circle_crop_v2(self, img):
        img = self.crop_image_from_gray(img)

        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = self.crop_image_from_gray(img)

        return img


    def load_ben_color(self, img, IMG_SIZE, sigmaX):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.circle_crop_v2(img)
        img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
        img = cv2.addWeighted (img, 4, cv2.GaussianBlur(img ,(0,0) ,sigmaX) ,-4 ,128)
            
        return img


# Get dataset directory
train_dir = os.path.join(os.getcwd(), "dataset", "train")
val_dir = os.path.join(os.getcwd(), "dataset", "val")
test_dir = os.path.join(os.getcwd(), "dataset", "test")


# Define the transform
transform = v2.Compose([
    CustomTransform(IMG_SIZE=224, sigmaX=10),
    v2.ToTensor(),  # Add more transforms as needed
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                        std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel)
])


# Create dataloaders
train_dataloader, test_dataloader, class_names = preprocessing.create_dataloaders(train_dir=train_dir,
                                                                                  test_dir=val_dir,
                                                                                  transform=transform,
                                                                                  num_workers=NUM_WORKERS,
                                                                                  # n_train_subset=10000,
                                                                                  # n_test_subset=1000,
                                                                                  batch_size=32) # set mini-batch size to 32
transform = CustomTransform(IMG_SIZE=256, sigmaX=1.5)

# Assuming you have a DataLoader or a loop over your dataset
for inputs, labels, folder_type in your_dataloader:
    for img, class_label in zip(inputs, labels):
        preprocessed_img = transform(img, folder_type, class_label)