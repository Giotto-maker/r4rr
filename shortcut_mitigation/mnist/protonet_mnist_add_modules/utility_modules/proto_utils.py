import torch
import threading


# * Builds random support sets from the given images and labels.
def get_random_classes(images, labels, n_support, n_classes=10):
    unique_classes = torch.unique(labels)
    assert len(unique_classes) == n_classes, f"There should be exactly {n_classes} unique classes."
        
    selected_images = []
    selected_labels = []

    for cls in unique_classes:
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        assert len(class_indices) >= n_support, f"Not enough samples for class {cls}"
        random_indices = torch.randperm(len(class_indices))[:n_support]
        selected_images.append(images[class_indices[random_indices]])
        selected_labels.append(labels[class_indices[random_indices]])

    if n_classes > 0:
        selected_images = torch.cat(selected_images)
        selected_labels = torch.cat(selected_labels)
    else:
        selected_images = torch.empty((0, 1, 28, 28))
        selected_labels = torch.empty((0, 1), dtype=torch.long)

    return selected_images, selected_labels


# * Returns the dataloader for the prototypical network (it contains the original prototypes with both support and query images)
def init_dataloader():
    dataloader_lock = threading.Lock()
    with dataloader_lock:
        proto_loader = torch.load('data/prototypes/proto_loader_dataset.pth')
    return proto_loader