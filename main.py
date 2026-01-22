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

from src.dataset import PigDataset, build_train_transforms, visualize_dataloader
from src.train import train, test, inference, evaluate_results_test, tta_prediction
import os
import mlflow
from torch.utils.data import WeightedRandomSampler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.models import get_model_and_transforms

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

        model_name = yaml_parser["model"]["type"]
        model, main_transform, weights_enum = get_model_and_transforms(
            model_name, 
            yaml_parser["num_classes"]
        )
        
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda")
        print(device)
        model.to(device)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable parameters: ", trainable)
        
        resume_itr = 0
        if yaml_parser["resume"] is not None and yaml_parser["resume"] != "None":
            resume_itr = yaml_parser["resume_itr"]
            print("Resuming from checkpoint: ", yaml_parser["resume"])
            model.load_state_dict(torch.load(yaml_parser["resume"]), strict=True)
            

        with mlflow.start_run(run_name=yaml_parser["run_name"]):
            mlflow.log_params(yaml_parser.get_flat_config())
            mlflow.log_artifact(args.config)

            if yaml_parser["task"] == "train":
                #Define dataloader
                if not yaml_parser["shuffle"]:
                    dataset = PigDataset(yaml_parser["train_ann"], yaml_parser["train_dir"], transform=build_train_transforms(yaml_parser, main_transform))
                    targets = dataset.targets
                    class_sample_count = np.unique(targets, return_counts=True)[1]
                    weight = 1. / class_sample_count
                    samples_weight = torch.tensor([weight[t] for t in targets])
                    
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
                    dataset = PigDataset(yaml_parser["train_ann"], yaml_parser["train_dir"], transform=build_train_transforms(yaml_parser, main_transform))
                    train_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], sampler=sampler, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=yaml_parser["drop_last"])
                else:
                    dataset = PigDataset(yaml_parser["train_ann"], yaml_parser["train_dir"], transform=build_train_transforms(yaml_parser, main_transform))
                    train_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=yaml_parser["shuffle"], pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=yaml_parser["drop_last"])
                
                dataset = PigDataset(yaml_parser["val_ann"], yaml_parser["val_dir"], transform=main_transform)
                val_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=False)
                dataset = PigDataset(yaml_parser["test_ann"], yaml_parser["test_dir"], transform=main_transform)
                test_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=False)
                
                device = torch.device("cuda")
                loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
                scaler = torch.cuda.amp.GradScaler()
                
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

                train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, scaler, device, yaml_parser["epoch"], output_dir, scheduler=scheduler, resume_itr=resume_itr)

            elif yaml_parser["task"] == "test":
                if yaml_parser["resume"] == None:
                    raise ValueError("No weights provided")
                
                dataset = PigDataset(yaml_parser["test_ann"], yaml_parser["test_dir"], transform=main_transform)
                test_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=False)
                
                loss_fn = nn.CrossEntropyLoss()
                device = torch.device("cuda")

                predictions, gt, loss = tta_prediction(test_loader, model, loss_fn, device)
                results = evaluate_results_test(predictions, gt)

                plt.figure(figsize=(6, 6))
                plt.imshow(results["cm"])
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.colorbar()
                mlflow.log_figure(plt.gcf(), "confusion_matrix_val.png")
                plt.close()

                # Log final test results (distinct from validation)
                print(results)
                mlflow.log_metric("final_test_loss", loss)
                mlflow.log_metric("final_test_ac", results["ca"])
                mlflow.log_metric("final_test_f1", results["f1"])


            elif yaml_parser["task"] == "inference":
                if yaml_parser["resume"] == None:
                    raise ValueError("No weights provided")
                    
                dataset = PigDataset(yaml_parser["test_ann"], yaml_parser["test_dir"], transform=main_transform, is_inference=True)
                test_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=False)

                device = torch.device("cuda")
                loss_fn = nn.CrossEntropyLoss()

                prob = False
                if yaml_parser.has_key("tta") and yaml_parser["tta"]:
                    predictions, row_ids, _ = tta_prediction(test_loader, model, loss_fn, device, prob=prob)
                else:
                    predictions, row_ids = inference(test_loader, model, device)

                results = pd.DataFrame([])
                new_rows = []
                for row in row_ids:
                    new_rows += row
                results["row_id"] = new_rows
                if prob:
                    for i in range(5):
                        results["class_"+str(i)] = torch.cat(predictions, dim=0)[:, i].cpu().numpy()
                else:
                    results["class_id"] = torch.cat(predictions).cpu().numpy()

                results.to_csv(os.path.join(yaml_parser["output_dir"], "predictions.csv"), index=False)
                print("Predictions saved in " + os.path.join(yaml_parser["output_dir"], "predictions.csv"))
            
                
            elif yaml_parser["task"] == "visualization":
                dataset = PigDataset(yaml_parser["train_ann"], yaml_parser["train_dir"], transform=build_train_transforms(yaml_parser))
                train_loader = DataLoader(dataset, batch_size=yaml_parser["batch_size"], shuffle=False, pin_memory=yaml_parser["pin_memory"], num_workers=yaml_parser["num_workers"], drop_last=yaml_parser["drop_last"])
                visualize_dataloader(train_loader, 10)
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

    #args = Namespace(config="configs/inference.yaml")
    #args = Namespace(config="configs/test.yaml")
    #args = Namespace(config="configs/train.yaml")
    #args = Namespace(config="configs/train_efficient_net.yaml")
    args = Namespace(config="configs/vit_dino/inference_vit_dino.yaml")

    main(args)
