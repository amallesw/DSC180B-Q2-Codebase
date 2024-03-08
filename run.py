import os
import numpy as np
import argparse

from config import config
from utils import (get_file_paths, generate_paths, get_feature_extractor,
                   get_model, split_dataset)
from datasets.pretrained_model_dataset import PretrainedModelDataset

def run(args, config):
    
    model_arch = args['model_arch']
    classifier = args['classifier']
    training_type = args['training_type']
    dataset_name = args['dataset_name']
    filepaths = get_file_paths(dataset_name)
    
    print(f"Loading {dataset_name} dataset...")
    dataset_path = filepaths["hdf5_file_path"]
    datasets = split_dataset(dataset_path, dataset_name, args["augment_data"])
    
    if classifier in ["CNN", "RNG"]:  # Adjust if "RNG" is not intended.
        final_str = f"Results for {classifier} on {dataset_name}"
        num_epochs = config['model']['CNN']['num_epochs']
    else:
        final_str = (f"Results for {model_arch} embeddings with {training_type} "
                     f"training using a {classifier} on {dataset_name}")
        num_epochs = config['model'][model_arch][classifier]['num_epochs']
    
        feature_extractor = get_feature_extractor(
            model_arch, config["model"][model_arch]["model_name"],
            config["batch_size"], training_type, 
            config["model"][model_arch]["unfreeze_layers"], config["device"])
        embeddings_data = {}
        
        for split in ["train", "val", "test"]:
            output_paths = generate_paths(args, split)
            try:
                embeddings = np.load(output_paths[0])
                labels = np.load(output_paths[1])
                print(f"Loaded {split} embeddings.")
            except FileNotFoundError:
                print(f"Extracting {split} embeddings...")
                embeddings, labels = feature_extractor.extract_and_save_embeddings(
                    datasets[split], *output_paths)
            
            embeddings_data[f"embeddings_{split}"] = embeddings
            embeddings_data[f"labels_{split}"] = labels
            
    model = get_model(args, config, labels=embeddings_data.get("labels_train"))
    train_loader, val_loader, test_loader = model.create_dataloaders(
        datasets if classifier in ["CNN", "RNG"] else embeddings_data)
    
    print("Training and validating...")
    model.train_and_validate(train_loader, val_loader, num_epochs)
#     print(f"Best validation accuracy: {best_val_acc:.2f}")
    
    print("Evaluating on test set...")
    test_metrics = model.evaluate_model(test_loader, model.emotion_labels)
    print(final_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the model."
    )
    parser.add_argument("--model_arch", type=str,
                        choices=['vit', 'resnet', 'efficientnet', 'none'],
                        default='none', help="Model architecture")
    parser.add_argument("--classifier", type=str,
                        choices=['NN', 'SVM', 'RF', 'CNN', 'RNG'],
                        required=True, help="Classifier type")
    parser.add_argument("--training_type", type=str,
                        choices=['full', 'partial', 'none'],
                        default='none', help="Training type for models")
    parser.add_argument("--dataset_name", type=str,
                        choices=['fer', 'dartmouth'], required=True,
                        help="Dataset name")
    parser.add_argument("--augment_data", action='store_true',
                        help="Augment data for training")

    args = parser.parse_args()

    # Normalize 'none' inputs to None
    args.model_arch = None if args.model_arch == 'none' else args.model_arch
    args.training_type = None if args.training_type == 'none' else args.training_type

    # Validation checks
    if args.classifier in ['CNN', 'RNG']:
        if args.model_arch is not None or args.training_type is not None:
            print("Error: --model_arch and --training_type should not be set "
                  "when classifier is CNN or RNG.")
            sys.exit(1)
    elif args.model_arch is None:
        print("Error: --model_arch is required for classifiers other than "
              "CNN or RNG.")
        sys.exit(1)
    elif args.training_type is None and args.classifier not in ['CNN', 'RNG']:
        print("Error: --training_type is required for pretrained model "
              "classifiers.")
        sys.exit(1)

    run(vars(args), config)