import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import numpy as np

from classifiers.base_classifier import BaseClassifier

class RandomForestModel(BaseClassifier):
    """
    A model wrapper for XGBoost, functioning as a Random Forest model 
    for classification tasks with methods for training, validation, and evaluation.

    Attributes:
        config (dict): Configuration settings for the model.
        num_classes (int): The number of classes in the classification task.
        model (xgb.Booster or None): The trained XGBoost model.
    """

    def __init__(self, config: dict, model_architecture: str, classifier: str, num_classes: int = 7):
        super().__init__(config, model_architecture, classifier)
        self.config = config
        self.num_classes = num_classes
        self.model = None  # Placeholder for the trained model
        
    def create_dataloaders(self, data: dict, train_size: float = 0.7, random_state: int = 42) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Prepares the data for training and evaluation, returning tuples of data loaders for train, validation, and test sets.

        Args:
            data (dict): A dictionary containing the datasets for training, validation, and testing.
            train_size (float, optional): Proportion of the dataset to include in the train split. Defaults to 0.7.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.

        Returns:
            Tuple[Tuple, Tuple, Tuple]: Tuples containing train, validation, and test sets.
        """
        X_train, y_train = data["embeddings_train"], data["labels_train"]
        X_val, y_val = data["embeddings_val"], data["labels_val"]
        X_test, y_test = data["embeddings_test"], data["labels_test"]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_and_validate(self, train_loader: Tuple, val_loader: Tuple, num_epochs: int) -> None:
        """
        Trains the model using XGBoost on the provided training set and validates it on the validation set.

        Args:
            train_loader (Tuple): The training data loader.
            val_loader (Tuple): The validation data loader.
            num_epochs (int): The number of epochs to train the model.
        """
        X_train, y_train = train_loader
        X_val, y_val = val_loader
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective': 'multi:softmax',
            'num_class': self.num_classes,
            'tree_method': 'gpu_hist',  # Adjust based on your setup
        }
        evals = [(dtrain, 'train'), (dval, 'eval')]
        evals_result = {}
        self.model = xgb.train(params, dtrain, num_boost_round=num_epochs, evals=evals,
                               early_stopping_rounds=self.model_config['early_stopping_rounds'],
                               evals_result=evals_result)
        
        return None

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], class_names: List[str]) -> None:
        """
        Plots the confusion matrix for the model predictions against true labels.

        Args:
            y_true (List[int]): True labels for the data.
            y_pred (List[int]): Predicted labels by the model.
            class_names (List[str]): Names of the classes for labels.
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Normalized Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')

        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self) -> None:
        """
        Plots the feature importances of the trained model.
        """
        importances = self.model.get_score(importance_type='weight')
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        keys = [item[0] for item in sorted_importances][:20]
        values = [item[1] for item in sorted_importances][:20]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=values, y=keys)
        plt.title('Feature Importances')
        plt.show()

    def evaluate_model(self, test_loader: Tuple, class_names: List[str]) -> Tuple:
        """
        Evaluates the model on the test set and plots evaluation metrics.

        Args:
            test_loader (Tuple): The test data loader.
            class_names (List[str]): Names of the classes for the classification report.

        Returns:
            Tuple: True labels and predicted labels.
        """
        X_test, y_test = test_loader
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        self.plot_confusion_matrix(y_test, y_pred, class_names)
        self.plot_feature_importances()
        
        return y_test, y_pred
