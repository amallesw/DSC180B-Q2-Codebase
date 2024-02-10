import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../data/FER/fer2013_resnet18_labels.npy')

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
from torch.optim import Optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
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

def create_dataloaders(data, labels=None, train_size: float = 0.7, random_state: int = 42, use_embeddings: bool = False):
    """
    DataLoader creation for both precomputed embeddings with labels and PyTorch Datasets.

    Parameters:
    - data: Embeddings as a numpy array or a PyTorch Dataset.
    - labels: Labels as a numpy array (used only if use_embeddings=True).
    - batch_size: Size of each batch for the DataLoader.
    - train_size: Proportion of the dataset to include in the train split.
    - random_state: Seed used by the random number generator for shuffling.
    - use_embeddings: Flag to indicate whether data consists of embeddings and labels (True) or a Dataset (False).

    Returns:
    - A dictionary containing DataLoader objects for 'train', 'val', and 'test' datasets.
    """
    print("Creating data loaders...")
    
    if use_embeddings:
        X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=1-train_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
    else:
        val_size, test_size = (1 - train_size) * 0.5, (1 - train_size) * 0.5
        train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

       
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

        
    return train_loader, val_loader, test_loader


def train_and_validate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: Optimizer, num_epochs: int = config["num_epochs"]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains and validates the model.

    Args:
        model (nn.Module): The model to train and validate.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: Lists containing training losses, training accuracies, validation losses, and validation accuracies for each epoch.
    """
    print("Training and Validating...")
    
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        total_loss, total_correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(config["device"]), labels.to(config["device"])

            optimizer.zero_grad()  
            output = model(images)
    
            if isinstance(output, torch.Tensor):
                logits = output
            else:  # Assuming it's an ImageClassifierOutput or similar
                logits = output.logits
    
            loss = criterion(logits, labels)
            loss.backward()        
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_correct / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validation
        model.eval()
        total_val_loss, total_val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config["device"]), labels.to(config["device"])
                output = model(images)
                
                if isinstance(output, torch.Tensor):
                    logits = output
                else:  # Assuming it's an ImageClassifierOutput or similar
                    logits = output.logits
                    
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total_val_correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_correct / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}")
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")
        print(f"Time taken: {epoch_duration:.2f} seconds")
        print("-" * 50)
        
    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_metrics(train_losses: List[float], val_losses: List[float], train_accuracies: List[float], val_accuracies: List[float]) -> None:
    """
    Plots training and validation losses and accuracies.

    Args:
        train_losses (List[float]): Training losses for each epoch.
        val_losses (List[float]): Validation losses for each epoch.
        train_accuracies (List[float]): Training accuracies for each epoch.
        val_accuracies (List[float]): Validation accuracies for each epoch.
    """
    plt.figure(figsize=(12, 5))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    return

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device to run the evaluation on.

    Returns:
        Tuple[List[int], List[int]]: Lists of true labels and predicted labels.
    """
    print("Evaluating...")
    model.eval()
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            if isinstance(output, torch.Tensor):
                logits = output
            else:  # Assuming it's an ImageClassifierOutput or similar
                logits = output.logits
    
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(predicted.cpu().numpy().flatten())

    return all_labels, all_predictions


def calculate_and_plot_metrics(all_labels: List[int], all_predictions: List[int]) -> np.ndarray:
    """
    Calculates and plots the evaluation metrics.

    Args:
        all_labels (List[int]): List of true labels.
        all_predictions (List[int]): List of predicted labels.

    Returns:
        None
    """
    test_accuracy = np.sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    print(f"Total labels: {len(all_labels)}, Total predictions: {len(all_predictions)}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    report = classification_report(all_labels, all_predictions, target_names=config["emotion_labels"])
    print(report)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config["emotion_labels"], yticklabels=config["emotion_labels"], ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=config["emotion_labels"], yticklabels=config["emotion_labels"], ax=ax2)
    ax2.set_title('Normalized Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()
    plt.show()
    
    return None