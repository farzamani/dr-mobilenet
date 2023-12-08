import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import torch.nn as nn
from time_code import Timer
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from PIL import Image
from typing import List, Tuple
import visualize
import farhad_preprocessing
from torchinfo import summary
from sklearn.metrics import confusion_matrix
import seaborn as sn

device = "cuda" if torch.cuda.is_available() else "cpu"

################### dataloaders
val_transform = v2.Compose([
#    v2.Resize((512,512)),
    v2.CenterCrop(384),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
])

#create dataloaders, i dont care about the training ones now
test_data = datasets.ImageFolder('dataret/farhad_preprocessed/test/', transform = val_transform)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                        shuffle=False, num_workers=2)

############ model
#weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
model = torchvision.models.mobilenet_v3_large().to(device)

# Get the length of class_names (one output unit for each class)
output_shape = 2

# modify last layer (classifier)
model.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1024, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.4),
    nn.Linear(in_features=1024, out_features=output_shape, bias=True),
).to(device)


## upload trained model 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

checkpoint = torch.load('models/model_6_mobilenet.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.to(device)


y_pred = []
y_true = []

# iterate over test data
for inputs,labels in testloader:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# print("y_pred =", y_pred)
# print("y_true =", y_true)

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# Calculate confusion matrix elements
for true_label, pred_label in zip(y_true, y_pred):
    if true_label == 1 and pred_label == 1:
        true_positive += 1
    elif true_label == 0 and pred_label == 0:
        true_negative += 1
    elif true_label == 0 and pred_label == 1:
        false_positive += 1
    elif true_label == 1 and pred_label == 0:
        false_negative += 1

# Display the confusion matrix
print("Confusion Matrix:")
print(f"True Positive: {true_positive}")
print(f"True Negative: {true_negative}")
print(f"False Positive: {false_positive}")
print(f"False Negative: {false_negative}")

# Calculate the confusion matrix manually
num_classes = max(max(y_true), max(y_pred)) + 1
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, pred_label in zip(y_true, y_pred):
    conf_matrix[true_label, pred_label] += 1

# Normalize the confusion matrix by all data
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum()

# Display the normalized confusion matrix as a heatmap
plt.imshow(conf_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix (by all data)')
plt.colorbar()

classes = [str(i) for i in range(num_classes)]
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted')
plt.ylabel('True')

# Add text annotations
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f'{conf_matrix_normalized[i, j]:.2f}', ha='center', va='center', color='black')

plt.show()
plt.savefig('results/conf_mat_mobile.png')
