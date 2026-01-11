import torchvision.models as models
import torchvision.transforms as T
import torch
from ultralytics import YOLO

val_aug = T.Compose([
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #T.ToTensor()
])

def get_model_and_transforms(model_name, num_classes, weights_str="DEFAULT", input_size=224):
    if model_name == "convnext_small":
        weights_enum = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = models.convnext_small(weights=weights_enum)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        print(weights_enum.transforms())
        preprocess = weights_enum.transforms()
        #preprocess = val_aug

        
    elif model_name == "efficientnet_v2_m":
        weights_enum = models.EfficientNet_V2_M_Weights.DEFAULT
        model = models.efficientnet_v2_m(weights=weights_enum)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        preprocess = weights_enum.transforms()
    
    # elif model_name == "yolo11_cls":
    #     model = model = YOLO('yolo11n-cls.pt').model
    #     in_features = model.head.fc.in_features
    #     model.head.fc = torch.nn.Linear(in_features, num_classes)
    #     transform = T.Compose([
    #         T.Resize((input_size, input_size)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    #     ])
        
    else:
        raise ValueError(f"Model {model_name} not supported in factory.")

    print(preprocess)
    return model, preprocess, weights_enum