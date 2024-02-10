import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../data/FER/fer2013_resnet18_labels.npy')

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from typing import List, Tuple

config = {
    'hdf5_path': "../../data/fer_2013_processed.h5",
    'fer2013_resnet18_embeddings_path': "../../data/FER/fer2013_resnet18_embeddings.npy",
    'fer2013_resnet18_labels_path': "../../data/FER/fer2013_resnet18_labels.npy",
    'model_name': "resnet18",
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 30,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'emotion_labels': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    'L2_regularization': 1e-4,
    'unfreeze_layers': ['layer4', 'fc'],
}

labels = np.load(data_path)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
config["class_weights"] = torch.tensor(class_weights, dtype=torch.float).to(config['device'])


def initialize_resnet_model(model_name: str = config["model_name"], num_classes: int = 7, unfreeze_layers: list = None, custom_model: nn.Module = None) -> Tuple[nn.Module, nn.Module, Optimizer]:
    """
    Initializes Resnet for training. This can be a pretrained model or a custom model for training on embeddings.
    """
    print("Initializing Resnet Model...")
    
    if custom_model:
        model = custom_model
    else:
        model = getattr(models, model_name)(pretrained=True)
        
        if unfreeze_layers:
            for name, param in model.named_parameters():
                param.requires_grad = name in unfreeze_layers or any(ul in name for ul in unfreeze_layers)
        else:
            for param in model.parameters():
                param.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(config['device'])
    criterion = nn.CrossEntropyLoss(weight=config.get('class_weights'))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('L2_regularization'))

    return model, criterion, optimizer

def extract_and_save_resnet_embeddings(dataset: Dataset, model_name: str, output_path_embeddings: str, output_path_labels: str):
    """
    Extracts embeddings from a dataset using a specified model and saves them.
    """
    device = config['device']
    
    model = getattr(models, model_name)(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    embeddings, labels = [], []
    with torch.no_grad():
        for images, label in loader:
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    output_dir = os.path.dirname(output_path_embeddings)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(output_path_embeddings, embeddings)
    np.save(output_path_labels, labels)
    
    
    
# Usage example, replace with actual Dataset and paths
# dataset = YourDatasetHere()
# extract_and_save_resnet_embeddings(dataset, "resnet18", "path/to/save/embeddings.npy", "path/to/save/labels.npy")