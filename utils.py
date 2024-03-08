import numpy as np
import torch
from typing import Tuple, Dict, Any
import h5py
from sklearn.utils.class_weight import compute_class_weight

# Import your feature extractor classes and model classes here
from models.feature_extractors import ResNetFeatureExtractor, VITFeatureExtractor, EfficientNetFeatureExtractor
from classifiers.torch_models import NN, SVMModel, CNNEncoder
from classifiers.xgboost_model import RandomForestModel
from datasets.pretrained_model_dataset import PretrainedModelDataset

def get_file_paths(dataset_name: str) -> Dict[str, str]:
    """
    Generates file paths for the given dataset name.
    
    Parameters:
        dataset_name (str): The name of the dataset ('fer', 'dartmouth').
    
    Returns:
        Dict[str, str]: A dictionary containing the 'data_path' and 'hdf5_file_path' for the given dataset.
    """
    return {
        'data_path': f"data/{dataset_name}/{dataset_name}48x48.csv",
        'hdf5_file_path': f"data/{dataset_name}/{dataset_name}_preprocessed.h5"
    }

def generate_paths(args: Dict, segment: str) -> Tuple[str, str]:
    """
    Generates paths for embeddings and labels based on dataset segmentation and augmentation status.
    
    Parameters:
        args (Dict): Arguments containing 'dataset_name', 'model_arch', 'training_type', and 'augment_data'.
        segment (str): The segment of the dataset ('train', 'val', or 'test').
    
    Returns:
        Tuple[str, str]: Paths for embeddings and labels files.
    """
    augmentation_suffix = 'augmented' if args["augment_data"] and segment == 'train' else 'original'
    output_path_embeddings = f"data/{args['dataset_name']}/{args['dataset_name']}_{args['model_arch']}" + \
    f"_{args['training_type']}_{segment}_{augmentation_suffix}_embeddings.npy"
    output_path_labels = f"data/{args['dataset_name']}/{args['dataset_name']}_{args['model_arch']}" + \
    f"_{args['training_type']}_{segment}_{augmentation_suffix}_labels.npy"
    return output_path_embeddings, output_path_labels

def get_feature_extractor(model_arch: str, model_name: str, batch_size: int, training_type: str, 
                          unfreeze_layers: list, device: torch.device) -> Any:
    """
    Returns the appropriate feature extractor instance based on the model architecture.
    
    Parameters:
        model_arch (str): The architecture of the model ('resnet', 'vit', 'efficientnet').
        model_name (str): The specific model name.
        batch_size (int): The batch size for processing.
        training_type (str): The type of training ('full', 'partial').
        unfreeze_layers (list): A list of layers to unfreeze for 'partial' training.
        device (torch.device): The device to run the model on.
    
    Returns:
        Any: An instance of the specified feature extractor.
    """
    if model_arch == "resnet":
        return ResNetFeatureExtractor(model_name, batch_size, training_type, unfreeze_layers, device)
    elif model_arch == "vit":
        return VITFeatureExtractor(model_name, batch_size, training_type, unfreeze_layers, device)
    elif model_arch == "efficientnet":
        return EfficientNetFeatureExtractor(model_name, batch_size, training_type, unfreeze_layers, device)
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

def get_model(args: Dict, config: Dict, labels: np.ndarray = None) -> Any:
    """
    Returns the appropriate model instance based on the arguments and configuration.
    
    Parameters:
        args (Dict): Arguments containing 'model_arch', 'classifier'.
        config (Dict): The configuration settings for the model.
        labels (np.ndarray, optional): The labels for computing class weights. Defaults to None.
    
    Returns:
        Any: An instance of the specified model.
    """
    model_arch = args["model_arch"]
    classifier = args["classifier"]
    input_size = (512 if model_arch == "resnet" else 768 if model_arch == "vit" 
                  else 1280 if model_arch == "efficientnet" else None)
    num_classes = 7

    if classifier == "NN":
        model = NN(config, model_arch, classifier, input_size, num_classes=num_classes)
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(config['device'])
        loss_function = torch.nn.CrossEntropyLoss(weight=weights_tensor)
        model.setup(model.model, loss_function, model.optimizer)
    elif classifier == "SVM":
        model = SVMModel(config, model_arch, classifier, input_size, num_classes=num_classes)
    elif classifier == "CNN":
        model = CNNEncoder(config, None, classifier, config["model"][classifier]["input_size"])
    else:  
        model = RandomForestModel(config, model_arch, classifier, num_classes=num_classes)
        
    return model

def split_dataset(hdf5_filename: str, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                  test_ratio: float = 0.15) -> Dict[str, np.ndarray]:
    """
    Splits the dataset into train, validation, and test indices.
    
    Parameters:
        hdf5_filename (str): The path to the HDF5 file containing the dataset.
        train_ratio (float, optional): The ratio of the dataset to use for training. Defaults to 0.7.
        val_ratio (float, optional): The ratio of the dataset to use for validation. Defaults to 0.15.
        test_ratio (float, optional): The ratio of the dataset to use for testing. Defaults to 0.15.
    
    Returns:
        Dict[str, np.ndarray]: A dictionary with keys 'train_indices', 'val_indices', and 'test_indices'.
    """
    with h5py.File(hdf5_filename, 'r') as file:
        images = file['images'][:]
        labels = file['labels'][:]
    total_images = len(labels)

    indices = np.arange(total_images)
    np.random.shuffle(indices)

    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }



def split_dataset(hdf5_filename: str, dataset_name: str, augment_data: bool, 
                  train_ratio: float = 0.7, val_ratio: float = 0.15, 
                  test_ratio: float = 0.15) -> Dict[str, np.ndarray]:
    """
    Splits the dataset into train, validation, and test indices.
    
    Parameters:
        hdf5_filename (str): The path to the HDF5 file containing the dataset.
        dataset_name (str): Name of dataset ("dartmouth", "fer").
        augment_data (bool): Whether to augment train data or not.
        train_ratio (float, optional): The ratio of the dataset to use for training. Defaults to 0.7.
        val_ratio (float, optional): The ratio of the dataset to use for validation. Defaults to 0.15.
        test_ratio (float, optional): The ratio of the dataset to use for testing. Defaults to 0.15.
    
    Returns:
        Dict[str, np.ndarray]: A dictionary with keys 'train_indices', 'val_indices', and 'test_indices'.
    """
    with h5py.File(hdf5_filename, 'r') as file:
        images = file['images'][:]
        labels = file['labels'][:]
        
    total_images = len(labels)
    indices = np.arange(total_images)
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)
    
    train_images, train_labels = images[indices[:train_end]], labels[indices[:train_end]]
    val_images, val_labels = images[indices[train_end:val_end]], labels[indices[train_end:val_end]]
    test_images, test_labels = images[indices[val_end:]], labels[indices[val_end:]]
    
    datasets = {
        "train": PretrainedModelDataset(train_images, train_labels, grayscale=(dataset_name=="fer"), 
                                augment_training=(augment_data)),
        "val": PretrainedModelDataset(val_images, val_labels, grayscale=(dataset_name=="fer"), 
                              augment_training=False),
        "test": PretrainedModelDataset(test_images, test_labels, grayscale=(dataset_name=="fer"), 
                               augment_training=False)
    }
    
    return datasets
    