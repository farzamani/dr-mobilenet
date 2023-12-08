from torchcam.methods import SmoothGradCAMpp
import torch.nn as nn
import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

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


cam_extractor = SmoothGradCAMpp(model)

import os
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Assuming read_image, normalize, resize, model, SmoothGradCAMpp, and overlay_mask functions are defined

# List of image paths
image_paths = [
    "dataret/farhad_preprocessed/test/1/0_222d0ac042b4.png", # 1 -->0
    "dataret/farhad_preprocessed/test/1/0_194814669fee.png",
    "dataret/farhad_preprocessed/test/0/0_f233638e0e90.png", # 0 --> 1
    "dataret/farhad_preprocessed/test/0/0_e1ab92228e60.png",
    "dataret/farhad_preprocessed/test/1/0_57469423a012.png", # 1 --> 1
    "dataret/farhad_preprocessed/test/1/0_59e5212f7139.png",
    "dataret/farhad_preprocessed/test/0/0_519c6e8f78dc.png",# 0 --> 0
    "dataret/farhad_preprocessed/test/0/0_537e5c578f40.png"
]

# Set the number of rows and columns for the subplot grid
num_rows = 2  # Adjust as needed
num_cols = 4  # Adjust as needed

# Create a subplot grid
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

for i, image_path in enumerate(image_paths):
    # Read the image
    img = read_image(image_path)
    
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Assuming SmoothGradCAMpp is defined before the loop
    
        # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)


    if num_rows > 1 and num_cols > 1:
        row_index = i // num_cols
        col_index = i % num_cols
        axs[row_index, col_index].imshow(result)
        axs[row_index, col_index].axis('off')
        if i == 0 or i == 1:
          axs[row_index, col_index].set_title(f"False Negative (FN)")
        if i == 2 or i == 3:
          axs[row_index, col_index].set_title(f"False Positive (FP)")
        if i == 4 or i == 5:
          axs[row_index, col_index].set_title(f"True Positive (TP)")
        if i == 6 or i ==7:
          axs[row_index, col_index].set_title(f"True Negative (TN)")
    else:
        axs[i].imshow(result)
        axs[i].axis('off')
        # put names to the images
        if i == 0 or i == 1:
          axs.set_title(f"False Negative (FN)")
        if i == 2 or i == 3:
          axs.set_title(f"False Positive (FP)")
        if i == 4 or i == 5:
          axs.set_title(f"True Positive (TP)")
        if i == 6 or i ==7:
           axs.set_title(f"True Negative (TN)")

    # Save the result
    result.save(os.path.join("results", os.path.basename(image_path).replace(".png", "_cam.png")))

# Save the entire figure with all subplots
plt.tight_layout()
plt.savefig("results/all_cams_yes_label_good.png")
plt.show()
