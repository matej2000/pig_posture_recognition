from torch.utils.data import DataLoader, Dataset
import pandas as pd
import ast
import os
from torchvision.io import decode_image
import torchvision.transforms.functional as F
from torchvision import transforms


class PigDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_data = pd.read_csv(csv_file, converters={"bbox": ast.literal_eval})
        self.img_dir = img_dir
        self.transform = transform

        self.targets = self.img_data["class_id"].values
    
    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.img_data.iloc[idx]["image_id"])
        label = self.img_data.iloc[idx]["class_id"]
        x1, y1, w, h = self.img_data.iloc[idx]["bbox"]

        img = decode_image(img_loc)
        img = F.crop(img, int(y1), int(x1), int(h), int(w))


        if self.transform:
            img = self.transform(img)
        return img, label

def build_train_transforms(cfg, weights):
    aug = []
    if "augmentation" not in cfg.config.keys():
        return weights.transforms()

    if "random_resized_crop" in cfg["augmentation"].keys():
        aug.append(
            transforms.RandomResizedCrop(
                size=cfg["augmentation"]["random_resized_crop"]["size"],
                scale=cfg["augmentation"]["random_resized_crop"]["scale"]
            )
        )

    if "horizontal_flip" in cfg["augmentation"].keys():
        aug.append(transforms.RandomHorizontalFlip(p=cfg["augmentation"]["horizontal_flip"]["p"]))

    if "color_jitter" in cfg["augmentation"].keys():
        aug.append(
            transforms.ColorJitter(**cfg["augmentation"]["color_jitter"])
        )

    aug.append(weights.transforms())
    return transforms.Compose(aug)
        
        

        