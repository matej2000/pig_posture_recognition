from src.yaml_parser import YamlParser
from argparse import Namespace
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights, convnext_small, ConvNeXt_Small_Weights

from src.dataset import PigDataset, build_train_transforms
from src.train import train, test
import os
import mlflow
from torch.utils.data import WeightedRandomSampler
import numpy as np

def loss_CE(pred, gt, device="cuda"):
    loss = nn.CrossEntropyLoss()
    return loss(pred["labels"], torch.tensor([x[0]["category_id"] for x in gt].to(device)))

def main(args, ) -> None:
    yaml_parser = YamlParser(args.config)
    #mlflow.pytorch.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow.set_experiment("Pig_Posture_Experiment")
    mlflow.set_experiment(yaml_parser["experiment_name"])
    mlflow.pytorch.autolog()

    if args.config is not None:
        with mlflow.start_run(run_name=yaml_parser["run_name"]):
            mlflow.log_params(yaml_parser.get_flat_config())
            mlflow.log_artifact(args.config)

            model = convnext_small(ConvNeXt_Small_Weights.DEFAULT)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, yaml_parser["num_classes"])
            main_transform = ConvNeXt_Small_Weights.DEFAULT.transforms()
            device = torch.device("cuda")
            model.to(device)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Number of trainable parameters: ", trainable)
            
            resume_itr = 0
            if yaml_parser["resume"] is not None and yaml_parser["resume"] != "None":
                print("Resuming from checkpoint: ", yaml_parser["resume"])
                model.load_state_dict(torch.load(yaml_parser["resume"]), strict=True)
                resume_itr = int(yaml_parser["resume"].split(".")[0].split("_")[-1]) + 1
                print(resume_itr)

            if yaml_parser["task"] == "train":
                #Define dataloader
                if not yaml_parser["shuffle"]:
                    # class_weights = torch.tensor([
                    #     0.000676,
                    #     0.000605,
                    #     0.000212,
                    #     0.003049,
                    #     0.000325
                    # ])
                    # Calculate weights
                    dataset = PigDataset(yaml_parser["train_ann"], yaml_parser["train_dir"], transform=build_train_transforms(yaml_parser, ConvNeXt_Small_Weights.DEFAULT))
                    targets = dataset.targets
                    class_sample_count = np.unique(targets, return_counts=True)[1]
                    weight = 1. / class_sample_count
                    samples_weight = torch.tensor([weight[t] for t in targets])
                    
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
                    dataset = PigDataset(yaml_parser["train_ann"], yaml_parser["train_dir"], transform=build_train_transforms(yaml_parser, ConvNeXt_Small_Weights.DEFAULT))
                    train_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], sampler=sampler, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=yaml_parser["drop_last"])
                else:
                    dataset = PigDataset(yaml_parser["train_ann"], yaml_parser["train_dir"], transform=build_train_transforms(yaml_parser, ConvNeXt_Small_Weights.DEFAULT))
                    train_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=yaml_parser["shuffle"], pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=yaml_parser["drop_last"])
                dataset = PigDataset(yaml_parser["val_ann"], yaml_parser["val_dir"], transform=main_transform)
                val_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=False)
                dataset = PigDataset(yaml_parser["test_ann"], yaml_parser["test_dir"], transform=main_transform)
                test_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=False)
                
                device = torch.device("cuda")
                loss_fn = nn.CrossEntropyLoss()
                
                # Load optimizer
                if not yaml_parser["optimizer"]:
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001)
                else:
                    print("Using optimizer from config file")
                    optimizer_class = getattr(optim, yaml_parser["optimizer"]["type"])
                    optimizer = optimizer_class(model.parameters(), **yaml_parser["optimizer"]["params"])
                
                if yaml_parser["resume_optimizer"] is not None and yaml_parser["resume_optimizer"] != "None":
                    print("Resuming optimizer from checkpoint: ", yaml_parser["resume_optimizer"])
                    optimizer.load_state_dict(torch.load(yaml_parser["resume_optimizer"]))

                if yaml_parser["scheduler"]:
                    scheduler_class = getattr(lr_scheduler, yaml_parser["scheduler"]["type"])
                    scheduler = scheduler_class(optimizer, **yaml_parser["scheduler"]["params"])

                # Create output dir if it does not exist
                output_dir = os.path.join(yaml_parser["output_dir"], yaml_parser["run_name"])
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device, yaml_parser["epoch"], output_dir, scheduler=scheduler, resume_itr=resume_itr)

            elif yaml_parser["task"] == "test":
                dataset = PigDataset(yaml_parser["test_ann"], yaml_parser["test_dir"], transform=main_transform)
                test_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=False)
                
                loss_fn = nn.CrossEntropyLoss()
                device = torch.device("cuda")

                predictions, gt, loss = test(test_loader, model, loss_fn, device)
            

            elif yaml_parser["task"] == "inference":
                raise NotImplementedError
            else:
                raise ValueError("Unknown task")
    else:
        raise ValueError("No config")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--task', '-t', type=str, default="train")
    args = parser.parse_args()

    args = Namespace(config="configs/test.yaml")

    main(args)
