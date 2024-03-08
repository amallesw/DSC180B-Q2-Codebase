import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
from typing import Tuple, List, Type

from classifiers.base_classifier import BaseClassifier
from classifiers.classifier_models import EmotionNN, EmotionSVM, SimpleCNN


class TorchModel(BaseClassifier):
    """A model wrapper for PyTorch-based models, providing methods for training, validation, and evaluation.

    Attributes:
        model (nn.Module): The PyTorch model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer for training the model.
        device (torch.device): The device (CPU or GPU) where the model will be trained.
        scheduler (optim.lr_scheduler): Learning rate scheduler.
    """

    def __init__(self, config: dict, model_architecture: str, classifier: str):
        super().__init__(config, model_architecture, classifier)
        self.model: Type[nn.Module] = None
        self.criterion: Type[nn.Module] = None
        self.optimizer: Type[optim.Optimizer] = None
        self.device = config['device']
        self.scheduler: Type[optim.lr_scheduler.ReduceLROnPlateau] = None

    def setup(self, model: Type[nn.Module], criterion: Type[nn.Module], optimizer: Type[optim.Optimizer]) -> None:
        """Initializes the model, criterion, optimizer, and scheduler with given parameters.

        Args:
            model (Type[nn.Module]): The PyTorch model to be used.
            criterion (Type[nn.Module]): The loss function.
            optimizer (Type[optim.Optimizer]): The optimizer.
        """
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

    def create_dataloaders(self, data: dict, train_size: float = 0.7, 
                           random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Creates DataLoader instances for training, validation, and testing datasets.

        Args:
            data (dict): A dictionary containing embeddings and labels for training, validation, and testing.
            train_size (float, optional): The proportion of the dataset to include in the train split. 
            Defaults to 0.7. random_state (int, optional): Controls the shuffling applied to the data before 
            applying the split. Defaults to 42.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
        """
        train_dataset = TensorDataset(torch.tensor(data["embeddings_train"], dtype=torch.float), 
                                      torch.tensor(data["labels_train"], dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(data["embeddings_val"], dtype=torch.float), 
                                    torch.tensor(data["labels_val"], dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(data["embeddings_test"], dtype=torch.float), 
                                     torch.tensor(data["labels_test"], dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], 
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], 
                                shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], 
                                 shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def train_and_validate(self, train_loader: DataLoader, val_loader: DataLoader, 
                           num_epochs: int) -> float:
        """Trains and validates the model, returning the best validation accuracy achieved.

        Args:
            train_loader (DataLoader): The DataLoader for the training data.
            val_loader (DataLoader): The DataLoader for the validation data.
            num_epochs (int): The number of epochs to train for.

        Returns:
            float: The best validation accuracy achieved.
        """
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        early_stopping_counter = 0
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            train_loss, train_accuracy = self._run_epoch(train_loader, is_training=True)
            val_loss, val_accuracy = self._run_epoch(val_loader, is_training=False)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            self.scheduler.step(val_loss)

            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
                  f"Time: {epoch_end_time - epoch_start_time:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Epoch {epoch+1}: No improvement in validation loss for {early_stopping_counter} epochs.")
                if early_stopping_counter >= self.model_config["early_stopping_rounds"]:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    break

        self._plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
        return None

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> Tuple[float, float]:
        """Runs a single epoch of training or validation.

        Args:
            loader (DataLoader): DataLoader for the current phase (train or validation).
            is_training (bool): True if the model is in training mode, False if in validation mode.

        Returns:
            Tuple[float, float]: Average loss and accuracy for the epoch.
        """
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(is_training):
                output = self.model(images)
                loss = self.criterion(output, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def evaluate_model(self, test_loader: DataLoader, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the model on the test set and prints out metrics.

        Args:
            test_loader (DataLoader): The DataLoader for the test data.
            class_names (List[str]): The names of the classes for printing the classification report.

        Returns:
            Tuple[np.ndarray, np.ndarray]: True and predicted labels for the test set.
        """
        self.model.eval()
        all_labels, all_predictions = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        self._calculate_and_plot_metrics(np.array(all_labels), np.array(all_predictions), class_names)
        return np.array(all_labels), np.array(all_predictions)

    def _plot_metrics(self, train_losses: List[float], val_losses: List[float], train_accuracies: List[float], 
                      val_accuracies: List[float]) -> None:
        """Plots training and validation loss and accuracy over epochs.

        Args:
            train_losses (List[float]): Training losses for each epoch.
            val_losses (List[float]): Validation losses for each epoch.
            train_accuracies (List[float]): Training accuracies for each epoch.
            val_accuracies (List[float]): Validation accuracies for each epoch.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def _calculate_and_plot_metrics(self, all_labels: np.ndarray, all_predictions: np.ndarray, 
                                    class_names: List[str]) -> None:
        """Calculates and plots metrics based on the model's performance on the test set.

        Args:
            all_labels (np.ndarray): The true labels.
            all_predictions (np.ndarray): The model's predictions.
            class_names (List[str]): The names of the classes for the classification report.
        """
        accuracy = np.mean(all_labels == all_predictions)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        cm = confusion_matrix(all_labels, all_predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        print(report)

        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, 
                                                                         average='weighted')
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

        sns.set(style="white")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, 
                    yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted Labels')
        axes[0].set_ylabel('True Labels')

        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, 
                    yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted Labels')
        axes[1].set_ylabel('True Labels')

        plt.show()

class NN(TorchModel):
    """
    A subclass of TorchModel for Neural Network based classification using EmotionNN architecture.
    """

    def __init__(self, config: dict, model_architecture: str, classifier: str, input_size: int, 
                 num_classes: int = 7, dropout_rate: float = 0.5):
        super().__init__(config, model_architecture, classifier)
        self.model = EmotionNN(input_size, num_classes, dropout_rate)
        self.setup_model()

    def setup_model(self) -> None:
        """
        Sets up the model with specified loss function and optimizer.
        """
        criterion = nn.CrossEntropyLoss(weight=self.model_config.get('class_weights', None))
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_config['learning_rate'],
                               weight_decay=self.model_config.get('L2_regularization', 0))
        self.setup(self.model, criterion, optimizer)
        
class SVMModel(TorchModel):
    """
    A subclass of TorchModel for SVM based classification using EmotionSVM architecture.
    """

    def __init__(self, config: dict, model_architecture: str, classifier: str, input_size: int, num_classes: int = 7):
        super().__init__(config, model_architecture, classifier)
        self.model = EmotionSVM(input_size, num_classes)
        self.setup_model()

    def setup_model(self) -> None:
        """
        Sets up the model with specified loss function and optimizer.
        """
        criterion = nn.MultiMarginLoss()  
        optimizer = optim.SGD(self.model.parameters(), lr=self.model_config['learning_rate'],
                              weight_decay=self.model_config.get('L2_regularization', 0))
        self.setup(self.model, criterion, optimizer)
        
class CNNEncoder(TorchModel):
    """
    A subclass of TorchModel for Convolutional Neural Network based classification using SimpleCNN architecture.
    """

    def __init__(self, config: dict, model_architecture: str, classifier: str, input_size: int, num_classes: int = 7):
        super().__init__(config, model_architecture, classifier)
        self.model = SimpleCNN(num_classes=num_classes)
        self.setup_model()

    def setup_model(self) -> None:
        """
        Sets up the model with specified loss function and optimizer.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.setup(self.model, criterion, optimizer)

    def create_dataloaders(self, datasets: dict, train_size: float = 0.7, 
                           random_state: int = 42) -> Tuple[DataLoader,DataLoader, DataLoader]:
        """
        Overrides TorchModel.create_dataloaders to accommodate datasets input directly as a dictionary.
        """
        train_dataset, val_dataset, test_dataset = datasets["train"], datasets["val"], datasets["test"]
        train_loader = DataLoader(train_dataset, batch_size=self.model_config["batch_size"], 
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.model_config["batch_size"], 
                                shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.model_config["batch_size"], 
                                 shuffle=False, num_workers=4)
        return train_loader, val_loader, test_loader
    
