from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
import os
import mlflow
import numpy as np

def evaluate_results(predictions, gt):
    gt2 = torch.cat(gt).cpu().numpy()
    predictions2 = torch.cat(predictions).cpu().numpy()
    f1 = f1_score(gt2, predictions2, average="macro")
    ca = np.mean(gt2 == predictions2)
    return f1, ca

def train_one_epoch(dataloader, model, loss_fn, optimizer, device, postprocessor):
    size = len(dataloader.dataset)
    
    batch = 0
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        
        pred = model(x)
        if postprocessor is not None:
            pred = postprocessor(pred)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # batch += 1

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(x)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device, epochs=1, output_dir="./output/", scheduler=None, resume_itr=0, postprocessor=None):
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer_type", type(optimizer).__name__)
    mlflow.log_param("learning_rate_init", optimizer.param_groups[0]["lr"])
    
    # Test
    predictions, gt, loss = test(val_loader, model, loss_fn, device, postprocessor=postprocessor)
    f1, ca = evaluate_results(predictions, gt)
    print(resume_itr)
    mlflow.log_metric("val_loss", loss, step=resume_itr-1)
    mlflow.log_metric("val_f1", f1, step=resume_itr-1)
    losses = [loss]
    f1s = [f1]
    learning_rates = [optimizer.param_groups[0]["lr"]]
    best_val_f1 = 0.0

    for t in range(resume_itr, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(train_loader, model, loss_fn, optimizer, device, postprocessor=postprocessor)
        predictions, gt, loss = test(val_loader, model, loss_fn, device, postprocessor=postprocessor)

        f1, ca = evaluate_results(predictions, gt)

        # MLflow log
        mlflow.log_metric("val_loss", loss, step=t)
        mlflow.log_metric("val_f1", f1, step=t)
        mlflow.log_metric("val_ac", ca, step=t)
        mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=t)
        
        # Save best epoch
        if f1 >= best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_epoch.pth"))
        
        #save last epoch
        torch.save(model.state_dict(), os.path.join(output_dir, f"last_epoch.pth"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, f"last_optimizer.pth"))


        losses.append(loss)
        f1s.append(f1)
        learning_rates.append(optimizer.param_groups[0]["lr"])
        print("Validation loss: ", losses)
        print("Validation f1: ", f1s)
        print("Learning rates: ", learning_rates)
    
    
    best_model = f1s.index(max(f1s))
    best_model_path = os.path.join(output_dir, "best_epoch.pth")
    print("Training complete. Evaluating best model on Test Set: " + str(best_model))
    model.load_state_dict(torch.load(best_model_path))
    
    test_preds, test_gt, test_loss = test(test_loader, model, loss_fn, device)
    test_f1, test_ca = evaluate_results(test_preds, test_gt)

    # Log final test results (distinct from validation)
    mlflow.log_metric("final_test_loss", test_loss)
    mlflow.log_metric("final_test_ac", test_ca)
    mlflow.log_metric("final_test_f1", test_f1)
    
    # Optional: Log the best model file as the final artifact
    mlflow.log_artifact(os.path.join(output_dir, f"last_epoch.pth"))
    mlflow.log_artifact(os.path.join(output_dir, f"best_epoch.pth"))
    mlflow.log_metric("best_epoch", best_model)

def test(dataloader, model, loss_fn, device, postprocessor=None):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    predictions = []
    gts = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            pred = model(x)

            test_loss += loss_fn(pred, y.to(device)).item()

            _, pred = torch.max(pred, 1)
            
            predictions.append(pred)
            gts.append(y)
            
    test_loss /= num_batches
    f1, ca = evaluate_results(predictions, gts)
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n F1: {f1:>8f} \n CA: {ca:>8f}")
    return predictions, gts, test_loss