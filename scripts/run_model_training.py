import argparse
import numpy as np
from torch import nn
from torch.utils.data import Dataset

import sys
import os

# Assuming this script is run from /home/amallesw/DSC180A-Q1-Codebase/scripts
# Add the parent directory (DSC180A-Q1-Codebase) to sys.path to resolve imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Now your imports should work
from model.base_models.train_evaluate_utils import create_dataloaders, train_and_validate, evaluate_model, plot_metrics, calculate_and_plot_metrics
from utils.resnet_utils import initialize_resnet_model
from model.custom_models.emotion_classifier_nn import EmotionClassifierNN
from model.base_models.custom_dataset import CustomDataset


# # Imports after adjusting sys.path
# from model.base_models.train_evaluate_utils import create_dataloaders, train_and_validate, evaluate_model, plot_metrics, calculate_and_plot_metrics
# from model.utils.resnet_utils import initialize_resnet_model
# from model.custom_models.emotion_classifier_nn import EmotionClassifierNN

def run_training_and_evaluation(model_type, training_method, num_epochs=10):
    
    custom_model = None
    if model_type == "resnet":
        custom_model = EmotionClassifierNN()
    elif model_type == "vit":
        # Initialize Vision Transformer model
        pass  # Replace with your Vision Transformer model initialization code
        
    # Load data based on training method
    if training_method == "embeddings":
        embeddings = np.load("../data/fer2013_resnet18_embeddings.npy")
        labels = np.load("../data/fer2013_resnet18_labels.npy")
        train_loader, val_loader, test_loader = create_dataloaders(data=embeddings, labels=labels, use_embeddings=True)
    else:
        # Load custom dataset or full data based on training method
        dataset = CustomDataset("../data/fer_2013_processed.h5")
        train_loader, val_loader, test_loader = create_dataloaders(dataset, use_embeddings=False)
        pass  # Replace with your code to load custom dataset or full data

    # Initialize model, criterion, and optimizer
    if training_method == "partial":
        unfreeze_layers = [...]  # Specify layers to unfreeze
        model, criterion, optimizer = initialize_resnet_model(unfreeze_layers=unfreeze_layers)
    else:
        model, criterion, optimizer = initialize_resnet_model()

    # Train and validate the model
    metrics = train_and_validate(model, train_loader, val_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)
    plot_metrics(*metrics)
    
    # Evaluate the model
    all_labels, all_predictions = evaluate_model(model, test_loader, device="cuda")
    calculate_and_plot_metrics(all_labels, all_predictions)
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run training and evaluation of emotion classifier model.")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "vit"],
                        help="Type of model to use: 'resnet' or 'vit'. Default is 'resnet'.")
    parser.add_argument("--training_method", type=str, default="full", choices=["full", "embeddings", "partial"],
                        help="Training method: 'full' (train entire model), 'embeddings' (train only last layers with embeddings), 'partial' (train specific layers). Default is 'full'.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training. Default is 10.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training_and_evaluation(args.model_type, args.training_method, args.num_epochs)
