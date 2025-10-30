from torch.utils.data import Dataset


class PrimitivesDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (Tensor): Tensor of shape [N, 3, 64, 64]
            labels (Tensor): Tensor of shape [N, 2] where:
                             - labels[:, 0] is the shape label  (0: square, 1: circle, 2: triangle)
                             - labels[:, 1] is the colour label (0: red, 1: yellow, 2: blue)
            transform: Optional transformation to apply to images.
        """
        self.images = images
        self.labels = labels  # shape [N, 2]
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        # Return shape label and colour label separately
        shape_label = self.labels[index, 0].long()
        color_label = self.labels[index, 1].long()
        if self.transform:
            image = self.transform(image)
        return image, shape_label.squeeze(), color_label.squeeze()

    def __len__(self):
        return len(self.images)
    


class SupervisedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (Tensor): Tensor of shape [N, 3, 64, 64]
            labels (Tensor): Tensor of shape [N, 6] where:
                             - labels[:, :3] are the shape labels  (0: square, 1: circle, 2: triangle)
                             - labels[:, 3:] are the colour labels (0: red, 1: yellow, 2: blue)
            transform: Optional transformation to apply to images.
        """
        self.images = images
        self.labels = labels  # shape [N, 6]
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index].long()
        if self.transform:
            image = self.transform(image)
        return image, label.squeeze(0)

    def __len__(self):
        return len(self.images)