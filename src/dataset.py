from torch.utils.data import DataLoader, Dataset
import pandas as pd
import ast
import os
from torchvision.io import decode_image
import torchvision.transforms.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

class PigDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_inference=False):
        self.img_data = pd.read_csv(csv_file, converters={"bbox": ast.literal_eval})
        self.img_dir = img_dir
        self.transform = transform
        self.is_inference = is_inference

        if not is_inference:
            self.targets = self.img_data["class_id"].values
    
    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.img_data.iloc[idx]["image_id"])
        if self.is_inference:
            label = self.img_data.iloc[idx]["row_id"]
        else:

            label = self.img_data.iloc[idx]["class_id"]
        
        x1, y1, w, h = self.img_data.iloc[idx]["bbox"]
        #size = w * h * 0.2
        x1 -= 50
        y1 -= 50
        w += 100
        h += 100
        img = decode_image(img_loc)
        img = F.crop(img, int(y1), int(x1), int(h), int(w))


        if self.transform:
            img = self.transform(img)
        return img, label

def build_train_transforms(cfg, weights=None):
    aug = []
    if "augmentation" not in cfg.config.keys():
        return weights.transforms()

    if "resize" in cfg["augmentation"].keys():
        aug.append(transforms.Resize((224, 224)))
    if "random_resized_crop" in cfg["augmentation"].keys():
        
        aug.append(
            transforms.RandomResizedCrop(
                size=cfg["augmentation"]["random_resized_crop"]["size"],
                scale=cfg["augmentation"]["random_resized_crop"]["scale"]
            )
        )

    if "horizontal_flip" in cfg["augmentation"].keys():
        aug.append(transforms.RandomHorizontalFlip(p=cfg["augmentation"]["horizontal_flip"]["p"]))
    
    if "random_perspective" in cfg["augmentation"].keys():
        aug.append(transforms.RandomPerspective(
            p=cfg["augmentation"]["random_perspective"]["p"],
            distortion_scale=cfg["augmentation"]["random_perspective"]["distortion_scale"])
        )

    if "color_jitter" in cfg["augmentation"].keys():
        aug.append(
            transforms.ColorJitter(**cfg["augmentation"]["color_jitter"])
        )

    if weights:
        aug.append(weights.transforms())
    return transforms.Compose(aug)
        
        
def visualize_dataloader(dataloader, num_images=10):
    images, labels = next(iter(dataloader))
    
    images = images[:num_images]
    labels = labels[:num_images]

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()

    class_names = ['Lat_Left', 'Lat_Right', 'Sitting', 'Standing', 'Sternal']

    # Define the constants used in your transform
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_images):
        # 1. Start with the raw tensor
        img = images[i]
        
        # 2. Denormalize: Reverse the (x - mean) / std operation
        # We do this while it's still a tensor for efficiency
        # img = img * std + mean
        
        # 3. Now clamp to [0, 1] and permute for plotting
        img = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i].item()])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
        