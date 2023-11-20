import torch
import os
import mlflow
import sys
import numpy as np
import traceback


from typing import Optional, Tuple
from tqdm import tqdm
from torchvision import models
from torchmetrics.classification import MultilabelF1Score

from src.models.custom_dataset import create_dataloaders, evaluate_disbalance


def run_iteration(images: torch.Tensor,
                  labels: torch.Tensor,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    outputs = model(images)
    loss_value = loss_fn(outputs.float(), labels.float())

    if optimizer:
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    return loss_value, outputs


def train_model(model_path: str,
                image_dir: str,
                logs_dir: str,
                num_classes: int,
                batch_size: int,
                num_epochs: int,
                val_proportion: float,
                test_proportion: float,):
    params = locals()

    mlflow.set_tracking_uri(logs_dir)
    mlflow.log_params(params)

    print('Creating dataloaders...')
    train_dataloader, val_dataloader, _ = \
        create_dataloaders(image_dir=image_dir,
                           batch_size=batch_size,
                           num_classes=num_classes,
                           val_proportion=val_proportion,
                           test_proportion=test_proportion)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
    model.head = torch.nn.Linear(1024, num_classes)

    if model_path:
        model = torch.load(model_path)

    model = model.to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    class_weights = evaluate_disbalance(train_dataloader)
    loss = torch.nn.BCEWithLogitsLoss(weight=class_weights.to(device))

    # Metrics
    f1_score_metric = MultilabelF1Score(num_labels=num_classes, average=None).to(device)
    f1_score_metric_micro = MultilabelF1Score(num_labels=num_classes, average='micro').to(device)

    print('Training model...')
    try:
        pbar_epochs = tqdm(range(num_epochs), desc="Epochs", ncols=150)
        for epoch in pbar_epochs:
            model.train()
            running_loss = 0.

            pbar_batches = tqdm(train_dataloader, desc="Training", leave=False, ncols=150)
            for images, labels in pbar_batches:
                images = images.to(device)
                labels = labels.to(device)

                loss_value, outputs = run_iteration(images, labels, model, loss, optimizer)
                running_loss += loss_value
                f1_score_metric.update(outputs, labels)
                f1_score_metric_micro.update(outputs, labels)

            train_loss = running_loss / len(train_dataloader)
            train_f1_score = f1_score_metric.compute()
            train_f1_score_micro = f1_score_metric_micro.compute()

            model.eval()
            running_loss = 0.

            f1_score_metric.reset()
            f1_score_metric_micro.reset()

            with torch.no_grad():
                for images, labels in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    loss_value, outputs = run_iteration(images, labels, model, loss)
                    running_loss += loss_value
                    f1_score_metric.update(outputs, labels)
                    f1_score_metric_micro.update(outputs, labels)

            val_loss = running_loss / len(val_dataloader)
            val_f1_score = f1_score_metric.compute()
            val_f1_score_micro = f1_score_metric_micro.compute()

            val_desc_str = (f'Val Loss: {round(float(val_loss), 2)} | '
                            f'Val F1 Score per Classes: {np.round(val_f1_score.cpu().numpy(), 3)} | '
                            f'Val F1 Score Micro: {np.round([val_f1_score_micro.cpu().numpy()], 3)}')

            pbar_epochs.set_postfix_str(val_desc_str)

            metrics = {
                "Train Loss": train_loss,
                "Train F1 Score Macro": np.mean(train_f1_score.cpu().numpy()),
                "Train F1 Score Micro": train_f1_score_micro.cpu().numpy(),
                "Val Loss": val_loss,
                "Val F1 Score Macro": np.mean(val_f1_score.cpu().numpy()),
                "Val F1 Score Micro": val_f1_score_micro.cpu().numpy(),
            }

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            f1_score_metric.reset()
            f1_score_metric_micro.reset()

    except KeyboardInterrupt:
        print('Training stopped by the user!')
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()
    finally:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        if exc_type is not None:
            print(f"\n{exc_type}: {exc_value}")
            traceback.print_tb(exc_traceback)
        else:
            print("No exceptions!")
            mlflow.pytorch.log_model(model, "model")
            mlflow.end_run()
            print('End of training!')
