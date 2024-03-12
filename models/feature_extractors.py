import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from transformers import ViTModel, ViTConfig
from typing import List, Tuple, Union


class FeatureExtractor:
    """
    Base class for feature extraction from pretrained models.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        training_type: str,
        unfreeze_layers: List[str],
        device: torch.device
    ):
        """
        Initializes the FeatureExtractor with model specifications.
        """
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.training_type: str = training_type
        self.unfreeze_layers: List[str] = unfreeze_layers
        self.device: torch.device = device

    def extract_and_save_embeddings(
        self,
        dataset: torch.utils.data.Dataset,
        output_path_embeddings: str,
        output_path_labels: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts embeddings from the given dataset and saves them to disk.
        """
        use_gpu = torch.cuda.is_available()
        num_workers = 4 if use_gpu else 0
        
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        embeddings, labels = [], []

        with torch.no_grad():
            for images, label in loader:
                images = images.to(self.device)
                output = self.extract_features(images)
                embeddings.append(output.cpu().numpy())
                labels.append(label.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save(output_path_embeddings, embeddings)
        np.save(output_path_labels, labels)
        print(f"Embeddings and labels saved to {output_path_embeddings} and {output_path_labels}.")
        return embeddings, labels

    def initialize_model(self) -> None:
        """
        Initializes pretrained model. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from images. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ResNetFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for ResNet18 models.
    """
    def __init__(self, model_name, batch_size, training_type, unfreeze_layers, device):
        super().__init__(model_name, batch_size, training_type, unfreeze_layers, device)
        self.model = self.initialize_model()

    def initialize_model(self) -> nn.Module:
        """
        Initializes and configures the ResNet model for feature extraction.
        """
        model = getattr(models, self.model_name)(pretrained=True)
        if self.training_type == "full":
            model.fc = nn.Identity()
        elif self.training_type == "partial":
            for param in model.parameters():
                param.requires_grad = False
            for layer_name in self.unfreeze_layers:
                for name, param in model.named_parameters():
                    if name.startswith(layer_name):
                        param.requires_grad = True
            model.fc = nn.Identity()
        return model.to(self.device)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from images using the ResNet model.
        """
        return self.model(images)


class VITFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for Vision Transformer models.
    """
    def __init__(self, model_name, batch_size, training_type, unfreeze_layers, device):
        super().__init__(model_name, batch_size, training_type, unfreeze_layers, device)
        self.model = self.initialize_model()

    def initialize_model(self) -> nn.Module:
        """
        Initializes and configures the Vision Transformer model for feature extraction.
        """
        config_vit = ViTConfig.from_pretrained(self.model_name)
        model = ViTModel.from_pretrained(self.model_name, config=config_vit)

        if self.training_type == "full":
            model.head = nn.Identity()
        elif self.training_type == "partial":
            for param in model.parameters():
                param.requires_grad = False
            for layer_name in self.unfreeze_layers:
                for name, param in model.named_parameters():
                    if name.startswith(layer_name):
                        param.requires_grad = True
            model.head = nn.Identity()
        return model.to(self.device)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from images using the Vision Transformer model.
        """
        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token representation


class EfficientNetFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for EfficientNet models.
    """
    def __init__(self, model_name, batch_size, training_type, unfreeze_layers, device):
        super().__init__(model_name, batch_size, training_type, unfreeze_layers, device)
        self.model = self.initialize_model()

    def initialize_model(self) -> nn.Module:
        """
        Initializes and configures the EfficientNet model for feature extraction.
        """
        model = getattr(models, self.model_name)(pretrained=True)

        if self.training_type == "full":
            model.classifier[1] = nn.Identity()
        elif self.training_type == "partial":
            for param in model.parameters():
                param.requires_grad = False

            for layer_name in self.unfreeze_layers:
                layer = dict([*model.named_children()])[layer_name]
                for param in layer.parameters():
                    param.requires_grad = True

            model.classifier[1] = nn.Identity()

        return model.to(self.device)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from images using the EfficientNet model.
        """
        return self.model(images)
