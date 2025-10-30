import torch 
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset


# * returns the dataloader for support and query images
def get_support_loader(support_images, support_labels, query_batch_size, debug=False):
    
    support_dataset = TensorDataset(support_images, support_labels)
    support_loader = DataLoader(support_dataset, batch_size=query_batch_size, shuffle=True)
    
    if debug:
        indices = torch.randperm(len(support_images))[:50]
        print("*********************************")
        print("CHECK SUPPORT SET")
        plot_support_primitives(support_images[indices], support_labels[indices])

    return support_loader


# * Reading the prototypes from file
def get_my_initial_prototypes(args):
    proto_images = torch.load('data/kand_annotations/pnet_proto/concept_prototypes.pt')
    proto_labels = torch.load('data/kand_annotations/pnet_proto/labels_prototypes.pt')
    print("Prototypical data loaded")
    print("Images: ", proto_images.shape)
    print("Labels: ", proto_labels.shape)

    if args.no_augmentations:
        print("Excluding Augmentations...")
        pairs = {(s, c): (proto_labels[:, 0] == s) & (proto_labels[:, 1] == c)
                for s in range(3) for c in range(3)}
        pair_indices = {k: v.nonzero(as_tuple=True)[0].tolist() for k, v in pairs.items()}
        selected_indices = []
        for s in range(3):
            for c in range(3):
                idxs = pair_indices[(s, c)]
                assert len(idxs) > 0, f"No samples found for pair ({s},{c})"
                chosen = random.choice(idxs)
                selected_indices.append(chosen)

        proto_images = proto_images[selected_indices]
        proto_labels = proto_labels[selected_indices]
        print("After excluding augmentations:")
        print("Images: ", proto_images.shape)
        print("Labels: ", proto_labels.shape)

    support_loader = get_support_loader(proto_images, proto_labels, query_batch_size=32, debug=False)
    print("Support loader created")
    return proto_images, proto_labels, support_loader


# * For prototypes computation at inference time
def get_random_classes(images, labels, n_support, n_classes=3):
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

    selected_images = torch.cat(selected_images)
    selected_labels = torch.cat(selected_labels)
    
    return selected_images, selected_labels

# * For prototypes computation at inference time (alternative - ensures all (shape,color) pairs are covered)
def get_random_classes_2(images, labels, n_support, allow_duplicates=True, args=None):
    """
    Select n groups of 9 images covering all (shape,color) pairs (0..2, 0..2).

    Args:
        images: tensor [N, ...]
        labels: tensor [N, 2]
        n_support: number of groups (total images = 9*n)
        allow_duplicates: if True and a pair has < n samples, we resample
                          with replacement (duplicates allowed).
                          If False we raise AssertionError.
        args: command line arguments.
    Returns:
        selected_images: tensor [9*n, ...]
        selected_labels: tensor [9*n, 2]
        selected_indices: list of chosen indices (in order)
    """
    if args is not None and hasattr(args, 'seed'):  seed = args.seed
    else:                                           seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    
    pair_indices = {(s, c): (labels[:, 0] == s).logical_and(labels[:, 1] == c)
                        .nonzero(as_tuple=True)[0].tolist()
                    for s in range(3) for c in range(3)}
    selected_indices = []
    for group in range(n_support):
        for s in range(3):
            for c in range(3):
                idxs = pair_indices[(s, c)]
                if len(idxs) == 0:
                    raise AssertionError(f"No samples found for pair ({s},{c})")
                if allow_duplicates:
                    chosen = random.choice(idxs)
                else:
                    assert len(idxs) >= 1,\
                        f"Not enough samples for pair ({s},{c}) to fill all groups"
                    chosen = random.choice(idxs)
                    idxs.remove(chosen)  # remove so not chosen again
                selected_indices.append(chosen)

    selected_images = images[selected_indices]
    selected_labels = labels[selected_indices]
    return selected_images, selected_labels


def plot_support_primitives(images, labels, num_images=50):

    # Calculate the number of rows and columns needed
    cols = 10
    rows = (num_images + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 1.5))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i].permute(1, 2, 0)
        label = labels[i].tolist()
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()