import h5py
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple

class PretrainedModelDataset(Dataset):
    """
    A dataset class for loading preprocessed images and labels from an HDF5 file, 
    designed for use with pretrained models. Supports optional data augmentation 
    and grayscale conversion.

    Attributes:
        hdf5_filename (str): Path to the HDF5 file containing the dataset.
        indices (np.ndarray): Array of indices to use from the dataset.
        grayscale (bool): Whether to convert images to grayscale.
        augment_training (bool): Whether to augment the data during training.
    """
#     def __init__(self, hdf5_filename: str, indices: np.ndarray, grayscale: bool = False, augment_training: bool = False):
    def __init__(self, images: np.ndarray, labels: np.ndarray, grayscale: bool = False, augment_training: bool = False):
        """
        Initializes the dataset object with the file path, indices, and other options.
        """
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.grayscale = grayscale
        self.augment_training = augment_training  # Indicates if augmentation is applied
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.transform = self.build_transforms()
        self.augment_transform = self.build_augment_transforms() if augment_training else None

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: The dataset size, doubled if augmentation is applied during training.
        """
        return len(self.labels) * 2 if self.augment_training else len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an item by its index. If augmentation is enabled and the dataset 
        is in training mode, it may return an augmented version of the image.
        
        Parameters:
            idx (int): The index of the item.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor 
            and its corresponding label tensor.
        """
        is_augmented = self.augment_training and idx >= len(self.labels)
        actual_idx = idx % len(self.labels)
        
        image = self.images[actual_idx]
        if is_augmented:
            image = self.augment_transform(image)
        else:
            image = self.transform(image)
        label = self.labels[actual_idx]
        return image, label

    def build_transforms(self) -> transforms.Compose:
        """
        Constructs the transformation pipeline for preprocessing images.
        
        Returns:
            transforms.Compose: The transformation pipeline.
        """
        transform_list = [transforms.ToPILImage(), transforms.Resize((224, 224)),
                          transforms.Grayscale(num_output_channels=3) if self.grayscale else transforms.Lambda(lambda x: x),
                          transforms.ToTensor()]
        return transforms.Compose(transform_list)

    def build_augment_transforms(self) -> transforms.Compose:
        """
        Constructs the augmentation transformation pipeline for training images.
        
        Returns:
            transforms.Compose: The augmentation transformation pipeline.
        """
        augment_list = [transforms.ToPILImage(), transforms.Resize((224, 224)),
                        transforms.Grayscale(num_output_channels=3) if self.grayscale else transforms.Lambda(lambda x: x),
                        transforms.Lambda(lambda img: self.apply_occlusion(img)),
                        transforms.ColorJitter(brightness=(0.3, 0.9)), transforms.ToTensor()]
        return transforms.Compose(augment_list)

    def apply_occlusion(self, image: Image.Image) -> Image.Image:
        """
        Applies occlusion to an image by obscuring detected eyes.
        
        Parameters:
            image (Image.Image): The image to occlude.
            
        Returns:
            Image.Image: The occluded image.
        """
        cv_image = np.array(image)
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_cascade.detectMultiScale(cv_image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        shrink_factor = 0.5

        for (ex, ey, ew, eh) in eyes:
            new_ew, new_eh = int(ew * shrink_factor), int(eh * shrink_factor)
            new_ex, new_ey = ex + (ew - new_ew) // 2, ey + (eh - new_eh) // 2
            cv2.rectangle(cv_image, (new_ex, new_ey), (new_ex + new_ew, new_ey + new_eh), (0, 0, 0), -1)

        return Image.fromarray(cv_image)
