import torch
import numpy as np

from torchmetrics.classification import MultilabelF1Score
from src.data.dataloaders import create_dataloaders


def eval_model(model_path: str,
               image_dir: str,
               num_classes: int,
               batch_size: int,
               val_proportion: float,
               test_proportion: float, ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path)
    print('Model has been loaded!')

    # Set dataloaders
    print('Creating dataloaders...')
    _, _, test_dataloader = create_dataloaders(image_dir=image_dir,
                                               batch_size=batch_size,
                                               num_classes=num_classes,
                                               val_proportion=val_proportion,
                                               test_proportion=test_proportion, )

    f1_score_metric = MultilabelF1Score(num_labels=num_classes, average=None).to(device)
    f1_score_metric_micro = MultilabelF1Score(num_labels=num_classes, average='micro').to(device)

    model.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            f1_score_metric(outputs, labels)
            f1_score_metric_micro(outputs, labels)

    test_f1_score = f1_score_metric.compute()
    test_f1_score_micro = f1_score_metric_micro.compute()

    test_desc_str = (f'Test F1 Score for classes: {test_f1_score.cpu().numpy()}\n'
                     f'Test F1 Score Macro: {np.mean(test_f1_score.cpu().numpy())}\n'
                     f'Test F1 Score Micro: {test_f1_score_micro.cpu().numpy()}'
                     )
    print(test_desc_str)
