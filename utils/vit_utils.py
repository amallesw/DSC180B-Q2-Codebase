import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTModel, ViTConfig
from your_project_name.config import config  # Adjust the import path according to your project structure

config = {
    'hdf5_path': "../../data/fer_2013_processed.h5",
    'fer2013_VIT_embeddings_path': "../../data/VIT/vit_embeddings.npy",
    'fer2013_VIT_labels_path': "../../data/VIT/vit_labels.npy",
    'model_name': 'google/vit-small-patch16-224',  # Assuming you're focusing on ViT small
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 10,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'emotion_labels': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    'L2_regularization': 1e-4,
    'unfreeze_layers': [
        'vit.encoder.layer.10', 
        'vit.encoder.layer.11', 
        'vit.layernorm', 
        'classifier.weight', 
        'classifier.bias'
    ],
}

def initialize_vit_model(model_name: str, num_classes: int, unfreeze_layers: list = None):
    """
    Initializes a Vision Transformer model for training, with the option for partial layer training.
    
    Args:
    - model_name (str): Identifier for the pre-trained model (e.g., 'google/vit-small-patch16-224').
    - num_classes (int): Number of classes for the classification head.
    - unfreeze_layers (list, optional): Specific layers to unfreeze for training. If None, all layers are trainable.
    
    Returns:
    - model (nn.Module): The initialized ViT model.
    - criterion (nn.Module): The loss function.
    - optimizer (Optimizer): The optimizer for training.
    """
    print("Initializing ViT Model...")
    
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)
    
    if unfreeze_layers is not None:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    model = model.to(config['device'])
    criterion = nn.CrossEntropyLoss(weight=config.get('class_weights', None))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=config['learning_rate'], 
                                  weight_decay=config.get('L2_regularization', 0))

    return model, criterion, optimizer

def extract_and_save_vit_embeddings(dataset, model_name, output_path_embeddings, output_path_labels):
    """
    Extracts embeddings from a dataset using a specified Vision Transformer model and saves them.
    """
    device = config['device']
    model = ViTModel.from_pretrained(model_name, config=ViTConfig.from_pretrained(model_name)).to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images, label = batch
            images = images.to(device)
            outputs = model(pixel_values=images)
            pooled_output = outputs.pooler_output
            embeddings.append(pooled_output.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path_embeddings), exist_ok=True)

    np.save(output_path_embeddings, embeddings)
    np.save(output_path_labels, labels)
    
    return embeddings, labels
