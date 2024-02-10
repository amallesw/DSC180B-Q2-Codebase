# custom_dataset.py
import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading data from an HDF5 file.

    Parameters:
    - hdf5_filename (str): Path to the HDF5 file containing the images and labels.
    - transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, hdf5_filename, transform=None):
        self.file = h5py.File(hdf5_filename, 'r')
        self.images = self.file['images']
        self.labels = self.file['labels']
        
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Gets the image and label at a specific index.
        
        Parameters:
        - idx (int): Index of the data point to retrieve.
        
        Returns:
        - Tuple[Tensor, Tensor]: A tuple containing the image and label tensors.
        """
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def close(self):
        """Closes the HDF5 file."""
        self.file.close()
