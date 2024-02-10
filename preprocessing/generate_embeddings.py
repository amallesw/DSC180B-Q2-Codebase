import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model', 'base_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from custom_dataset import CustomDataset
from resnet_utils import extract_and_save_resnet_embeddings
from vit_utils import extract_and_save_vit_embeddings
import torch

filepaths = {
    'resnet': {
        'model_name': "resnet18",
        'embeddings_path': "../../data/FER/fer2013_resnet18_embeddings.npy",
        'labels_path': "../../data/FER/fer2013_resnet18_labels.npy",
    },
    'vit': {
        'model_name': "google/vit-base-patch16-224-in21k",
        'embeddings_path': "../../data/VIT/vit_embeddings.npy",
        'labels_path': "../../data/VIT/vit_labels.npy",
    }
}

def main(model_type):
    config = filepaths[model_type]
    dataset = CustomDataset(config["hdf5_path"])

    if model_type == 'resnet':
        extract_and_save_embeddings = extract_and_save_resnet_embeddings
    elif model_type == 'vit':
        extract_and_save_embeddings = extract_and_save_vit_embeddings
    else:
        raise ValueError("Invalid model type specified. Choose either 'resnet' or 'vit'.")

    embeddings, labels = extract_and_save_embeddings(dataset, config["model_name"],
                                                     config["embeddings_path"],
                                                     config["labels_path"])
    print(f"{model_type.upper()} embeddings have been successfully extracted and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for a specified model type.')
    parser.add_argument('--model_type', type=str, help="Model type for generating embeddings. Options: 'resnet', 'vit'", required=True)
    args = parser.parse_args()

    main(args.model_type)