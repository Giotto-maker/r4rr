import torch
from torch.utils.data import Dataset, DataLoader


class MNISTAugDataset(Dataset):
    def __init__(self, images, labels, hide_labels=None, transform=None):
        # Store original dataset size
        self.original_length = labels.size(0)

        if hide_labels:
            # Create a boolean mask using torch.isin (available in recent versions of PyTorch)
            mask = ~torch.isin(labels, torch.tensor(hide_labels))
            # Use tensor indexing to filter both images and labels
            self.images = images[mask]
            self.labels = labels[mask]
            
            # Ensure the channel dimension is preserved
            if self.images.dim() == 3:  # If shape is [batch_filtered, 28, 28]
                self.images = self.images.unsqueeze(1)  # Add back the channel dim to get [batch_filtered, 1, 28, 28]
            if self.labels.dim() == 1:  # If shape is [batch_filtered]
                self.labels = self.labels.unsqueeze(1)  # Add back the channel dim to get [batch_filtered, 1]

            self.num_filtered = self.original_length - mask.sum().item()
        else:
            self.images = images
            self.labels = labels
            self.num_filtered = 0
        
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label.squeeze()

    def __len__(self):
        return len(self.labels)
    

# Define an empty dataset
class EmptyDataset(Dataset):
    def __len__(self):
        return 0  # No samples

    def __getitem__(self, index):
        raise IndexError("This dataset is empty")