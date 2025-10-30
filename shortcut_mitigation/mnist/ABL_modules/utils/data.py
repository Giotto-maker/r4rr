import torch
from torch.utils.data import Dataset

# * dataset to store PRETRAINING data
class MNISTAugDatasetPretraining(Dataset):
    """
    A custom PyTorch Dataset for MNIST images with optional transformations.

    Args:
        images (Tensor or ndarray): The dataset images.
        labels (Tensor or ndarray): The corresponding labels for the images.
        hide_labels (optional): Unused parameter, reserved for future use.
        transform (callable, optional): Optional transform to be applied on an image.

    Methods:
        __getitem__(index): Returns the (transformed) image and squeezed label at the specified index.
        __len__(): Returns the total number of samples in the dataset.
    """
    def __init__(self, images, labels, hide_labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label.squeeze()

    def __len__(self):
        return len(self.labels)
    

# * dataset to store SUPERVISIONS to guide TRAINING
class MNISTAugDatasetTraining(Dataset):
    def __init__(self, images, labels, concepts):
        self.images = images
        self.labels = labels
        self.concepts = concepts

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        concept = self.concepts[index]
        return image, label, concept

    def __len__(self):
        return len(self.labels)


def dataloader_to_list(dataloader):
    """
    Converts a PyTorch DataLoader for the MNIST shortcut mitigation dataset into separate lists of left/right digit images, concept labels, and digit labels.
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader yielding batches of (images, labels, concepts), where
            - images: Tensor of shape [batch_size, 1, 28, 56], concatenated left and right digit images,
            - labels: Tensor of shape [batch_size], digit labels,
            - concepts: Tensor of shape [batch_size, 2], concept labels.
    Returns:
        tuple: (X_list, gt_pseudo_list, Y_list)
            - X_list (list): List of [left_digit, right_digit] image tensors for each sample.
            - gt_pseudo_list (list): List of concept label lists for each sample.
            - Y_list (list): List of digit labels for each sample.
    Raises:
        AssertionError: If input tensors do not match expected shapes.
    """
    X_list, gt_pseudo_list, Y_list = [], [], []
    for batch in dataloader:
        images, labels, concepts = batch
        batch_size = images.shape[0]
        assert images.shape == torch.Size([batch_size, 1, 28, 56]), \
            f"Expected tensor shape [{batch_size}, 1, 28, 56], got {images.shape}"
        assert labels.shape == torch.Size([batch_size]), \
            f"Expected tensor shape [{batch_size}], got {labels.shape}"
        assert concepts.shape == torch.Size([batch_size, 2]), \
            f"Expected tensor shape [{batch_size}, 2], got {concepts.shape}"
        
        left_digits = images[:, :, :, :28]
        right_digits = images[:, :, :, 28:]

        for i in range(batch_size):
            X_list.append([left_digits[i], right_digits[i]])
            gt_pseudo_list.append(concepts[i].tolist())
            Y_list.append(labels[i].item())
            
    return X_list, gt_pseudo_list, Y_list


def print_data_format(dataset_train, dataset_val):
    print(f"Both train_data and test_data consist of 3 components: X, gt_pseudo_label, Y")
    print()
    train_X, train_gt_pseudo_label, train_Y = dataset_train
    print(
        f"Length of X, gt_pseudo_label, Y in train_data: "
        + f"{len(train_X)}, {len(train_gt_pseudo_label)}, {len(train_Y)}"
    )
    test_X, test_gt_pseudo_label, test_Y = dataset_val
    print(
        f"Length of X, gt_pseudo_label, Y in test_data: "
        + f"{len(test_X)}, {len(test_gt_pseudo_label)}, {len(test_Y)}"
    )
    print()

    X_0, gt_pseudo_label_0, Y_0 = train_X[0], train_gt_pseudo_label[0], train_Y[0]
    print(
        f"X is a {type(train_X).__name__}, "
        + f"with each element being a {type(X_0).__name__} "
        + f"of {len(X_0)} {type(X_0[0]).__name__} whose shape is {X_0[0].shape}."
    )
    print(
        f"gt_pseudo_label is a {type(train_gt_pseudo_label).__name__}, "
        + f"with each element being a {type(gt_pseudo_label_0).__name__} "
        + f"of {len(gt_pseudo_label_0)} {type(gt_pseudo_label_0[0]).__name__}."
    )
    print(f"Y is a {type(train_Y).__name__}, " + f"with each element being an {type(Y_0).__name__}.")


# * Yields an annotated training data example (concepts and label) given two annotated digits
def merge_mnist_pairs(images, labels):
    assert images.ndim == 4 and images.shape[1:] == (1, 28, 28), \
        f"Expected image shape (N, 1, 28, 28), got {images.shape}"
    assert labels.ndim == 2 and labels.shape[1] == 1, \
        f"Expected label shape (N, 1), got {labels.shape}"

    N = images.size(0)
    if N % 2 != 0:
        images = images[:-1]
        labels = labels[:-1]
        N -= 1

    merged_images = torch.cat([images[0::2], images[1::2]], dim=3)
    flat_labels = labels.squeeze(1)
    concepts = torch.stack([flat_labels[0::2], flat_labels[1::2]], dim=1)
    labels = concepts.sum(dim=1)

    return merged_images, concepts, labels