import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from .data_augmentations import ( 
    get_no_rotations, 
    get_no_translations, 
    get_no_scaled, 
    get_no_elastic_transformations, 
    get_no_noisy_transformations, 
    get_rotated, 
    get_translated, 
    get_scaled, 
    get_elastic_transformations, 
    get_noisy_transformations
)
from ..utility_modules.plotting import plot_digits
from .other_data_processing import apply_deskew


# * Selects 10 initial prototypes from the given dataloader.
def choose_initial_prototypes(dataloader, debug=False):
    """
    Selects initial prototypes from the given dataloader.
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        debug (bool, optional): If True, displays debug information including images and annotations. Defaults to False.
    Returns:
        torch.utils.data.DataLoader: DataLoader containing the selected prototype data.
    The function iterates through the dataloader and selects pairs of prototypes based on their annotations.
    It ensures that there are no duplicate prototypes in the chosen and inferred prototypes lists.
    The process stops once 10 unique prototypes have been selected.
    """
    chosen_prototypes = []   # right-hand side of MNIST pairs
    inferred_prototypes = [] # left-hand side of MNIST pairs
    proto_data = []          # all pairs

    for i, data in enumerate(dataloader):
        image, sum_label, annotation = data
        annotation = annotation.tolist()
        
        if len(chosen_prototypes) == 10: 
            print("Chosen prototypes: ", chosen_prototypes)
            print("Inferred prototypes: ", inferred_prototypes)
            break

        ann_idx = i % 32
        # ^ annotation[ann_idx][1] should be derived from label
        if annotation[ann_idx][0] not in chosen_prototypes and annotation[ann_idx][1] not in inferred_prototypes:
            chosen_prototypes.append(annotation[ann_idx][0])
            inferred_prototypes.append(annotation[ann_idx][1])
            proto_data.append( (image[ann_idx], 
                torch.tensor( [annotation[ann_idx][0], annotation[ann_idx][1]]) )
            )

    assert len(chosen_prototypes) == 10
    assert len(chosen_prototypes) == len(set(chosen_prototypes)), "Duplicates found in chosen_prototypes"
    assert len(inferred_prototypes) == len(set(inferred_prototypes)), "Duplicates found in inferred_prototypes"
    
    proto_loader = DataLoader(proto_data, batch_size=1)

    if debug:
        for _, data in enumerate(proto_loader):
            image, annotations = data

            assert isinstance(image, torch.Tensor), "Image is not a tensor"
            assert isinstance(annotations, torch.Tensor), "Annotations are not a tensor"
            assert image.shape == (1, 1, 28, 56), f"Image shape is not 1x1x28x56, but {image.shape}"
            assert annotations.shape == (1, 2), f"Annotations shape is not 1x2, but {annotations.shape}"

            print("Image's annotations: ", annotations[0])
            plt.imshow(image[0].numpy().reshape(28, 56), cmap='gray')
            plt.show()

    os.makedirs('data/prototypes', exist_ok=True)
    torch.save(proto_loader, 'data/prototypes/proto_loader_dataset.pth')
    return proto_loader


# * Returns the support and query set from the chosen prototyes in the dataloader.
def get_original_support_query_set(dataloader, debug=False):
    
    support_images = torch.zeros((10, 1, 28, 28))
    query_images = torch.zeros((10, 1, 28, 28))
    support_labels = torch.zeros((10, 1), dtype=torch.long)
    query_labels = torch.zeros((10, 1), dtype=torch.long)
    count_check = 0

    for image_pair, labels in dataloader:
        support_images[count_check] = image_pair[:, :, :, :28]    # Left-side digits (support set)
        query_images[count_check] = image_pair[:, :, :, 28:]      # Right-side digits (query set)
        support_labels[count_check] = labels[:, 0]  # Corresponding labels for left-side digit
        query_labels[count_check] = labels[:, 1]    # Corresponding labels for right-side digit
        
        if debug:
            print("Support set label: ", support_labels[count_check].item())
            plt.imshow(support_images[count_check].numpy().reshape(28, 28), cmap='gray')
            plt.show()
        
        count_check += 1
        assert count_check <= 10, "Support set has more than 10 images"    

    return support_images, support_labels, query_images, query_labels


# * Returns the augmented support and query set from the dataloader.
def get_augmented_support_query_set(dataloader, no_augmentations=False, debug=False):
    support_images, support_labels, query_images, query_labels = get_original_support_query_set(dataloader)
    support_images = apply_deskew(support_images)
    query_images = apply_deskew(query_images)
    if no_augmentations:
        return support_images, support_labels, query_images, query_labels, 0
    if debug:
        print("support_images_rot dtype: ", support_images.dtype)
        print("support_images_rot min value: ", support_images.min().item())
        print("support_images_rot max value: ", support_images.max().item())
        print("support_images_rot shape: ", support_images.shape)

    # ^ Initialize augmented tensors
    no_rot = get_no_rotations()
    print("Number of rotations: ", no_rot)
    no_trn = get_no_translations()
    print("Number of translations: ", no_trn)
    no_sld = get_no_scaled() 
    print("Number of scalings: ", no_sld)
    no_def = get_no_elastic_transformations()
    print("Number of elastic transformations: ", no_def)
    no_nsy = get_no_noisy_transformations()
    print("Number of noising transformations: ", no_nsy)

    no_aug = no_rot + no_trn + no_sld + no_def + no_nsy

    support_images_aug = torch.zeros((no_aug, 1, 28, 28))
    support_labels_aug = torch.zeros((no_aug, 1), dtype=torch.long)
    query_images_aug = torch.zeros((no_aug, 1, 28, 28))
    query_labels_aug = torch.zeros((no_aug, 1), dtype=torch.long)

    # ^ Rotations with labels
    support_images_rot, support_labels_rot = get_rotated(support_images, support_labels)
    query_images_rot, query_labels_rot = get_rotated(query_images, query_labels)
    support_images_rot = support_images_rot.view(-1, 1, 28, 28)
    query_images_rot = query_images_rot.view(-1, 1, 28, 28)

    if debug:
        print("Original support images")
        plot_digits(support_images, support_labels)
        print("Applied the rotation transformation to the support images")
        plot_digits(support_images_rot[-20:], support_labels_rot[-20:])
        print("Original query images")
        plot_digits(query_images, query_labels)
        print("Applied the rotation transformation to the query images")
        plot_digits(query_images_rot[-20:], support_labels_rot[-20:])

    assert support_images_rot.shape == (no_rot, 1, 28, 28), \
        f"Shape of support_images_rot is not {no_rot}x1x28x28, but {support_images_rot.shape}"
    assert query_images_rot.shape == (no_rot, 1, 28, 28), \
        f"Shape of query_images_rot is not {no_rot}x1x28x28, but {query_images_rot.shape}"
    assert support_labels_rot.shape == (no_rot, 1), \
        f"Shape of support_labels_rot is not {no_rot}x1, but {support_labels_rot.shape}"
    assert query_labels_rot.shape == (no_rot, 1), \
        f"Shape of query_labels_rot is not {no_rot}x1, but {query_labels_rot.shape}"
        
    support_images_aug[:no_rot] = support_images_rot
    support_labels_aug[:no_rot] = support_labels_rot
    query_images_aug[:no_rot] = query_images_rot
    query_labels_aug[:no_rot] = query_labels_rot

    if debug:
        print("support_images_rot dtype: ", support_images_rot.dtype)
        print("support_images_rot min value: ", support_images_rot.min().item())
        print("support_images_rot max value: ", support_images_rot.max().item())
        print("support_images_rot shape: ", support_images_rot.shape)

    # ^ Translations with labels
    support_images_trn, support_labels_trn = get_translated(support_images, support_labels)
    query_images_trn, query_labels_trn = get_translated(query_images, query_labels)
    support_images_trn = support_images_trn.view(-1, 1, 28, 28)
    query_images_trn = query_images_trn.view(-1, 1, 28, 28)

    if debug: 
        print("Original support images")
        plot_digits(support_images, support_labels)
        print("Applied the translation transformation to the support images")
        plot_digits(support_images_trn[-20:], support_labels_trn[-20:])
        print("Original query images")
        plot_digits(query_images, query_labels)
        print("Applied the translation transformation to the query images")
        plot_digits(query_images_trn[-20:], support_labels_trn[-20:])

    assert support_images_trn.shape == (no_trn, 1, 28, 28), \
        f"Shape of support_images_trn is not {no_trn}x1x28x28, but {support_images_trn.shape}"
    assert query_images_trn.shape == (no_trn, 1, 28, 28), \
        f"Shape of query_images_trn is not {no_trn}x1x28x28, but {query_images_trn.shape}"
    assert support_labels_trn.shape == (no_trn, 1), \
        f"Shape of support_labels_trn is not {no_trn}x1, but {support_labels_trn.shape}"
    assert query_labels_trn.shape == (no_trn, 1), \
        f"Shape of query_labels_trn is not {no_trn}x1, but {query_labels_trn.shape}"
    
    support_images_aug[no_rot:no_rot + no_trn] = support_images_trn
    support_labels_aug[no_rot:no_rot + no_trn] = support_labels_trn
    query_images_aug[no_rot:no_rot + no_trn] = query_images_trn
    query_labels_aug[no_rot:no_rot + no_trn] = query_labels_trn

    if debug:
        print("support_images_trn dtype: ", support_images_trn.dtype)
        print("support_images_trn min value: ", support_images_trn.min().item())
        print("support_images_trn max value: ", support_images_trn.max().item())
        print("support_images_trn shape: ", support_images_trn.shape)

    # ^ Scalings with labels
    support_images_sld, support_labels_sld = get_scaled(support_images, support_labels)
    query_images_sld, query_labels_sld = get_scaled(query_images, query_labels)
    support_images_sld = support_images_sld.view(-1, 1, 28, 28)
    query_images_sld = query_images_sld.view(-1, 1, 28, 28)

    if debug: 
        print("Original support images")
        plot_digits(support_images, support_labels)
        print("Applied the scaling transformation to the support images")
        plot_digits(support_images_sld[-20:], support_labels_sld[-20:])
        print("Original query images")
        plot_digits(query_images, query_labels)
        print("Applied the scaling transformation to the query images")
        plot_digits(query_images_sld[-20:], support_labels_sld[-20:])

    assert support_images_sld.shape == (no_sld, 1, 28, 28), \
        f"Shape of support_images_sld is not {no_sld}x1x28x28, but {support_images_sld.shape}"  
    assert query_images_sld.shape == (no_sld, 1, 28, 28), \
        f"Shape of query_images_sld is not {no_sld}x1x28x28, but {query_images_sld.shape}"
    assert support_labels_sld.shape == (no_sld, 1), \
        f"Shape of support_labels_sld is not {no_sld}x1, but {support_labels_sld.shape}"
    assert query_labels_sld.shape == (no_sld, 1), \
        f"Shape of query_labels_sld is not {no_sld}x1, but {query_labels_sld.shape}"
    
    support_images_aug[no_rot + no_trn:no_rot + no_trn + no_sld] = support_images_sld
    support_labels_aug[no_rot + no_trn:no_rot + no_trn + no_sld] = support_labels_sld
    query_images_aug[no_rot + no_trn:no_rot + no_trn + no_sld] = query_images_sld
    query_labels_aug[no_rot + no_trn:no_rot + no_trn + no_sld] = query_labels_sld

    if debug:
        print("support_images_sld dtype: ", support_images_sld.dtype)
        print("support_images_sld min value: ", support_images_sld.min().item())
        print("support_images_sld max value: ", support_images_sld.max().item())
        print("support_images_sld shape: ", support_images_sld.shape)

    # ^ Elastic transformations with labels
    support_images_def, support_labels_def = get_elastic_transformations(support_images, support_labels)
    query_images_def, query_labels_def = get_elastic_transformations(query_images, query_labels)
    support_images_def = support_images_def.view(-1, 1, 28, 28)
    query_images_def = query_images_def.view(-1, 1, 28, 28)

    if debug:
        print("Original support images")
        plot_digits(support_images, support_labels)
        print("Applied the elastic transformation to the support images")
        plot_digits(support_images_def[-20:], support_labels_def[-20:])
        print("Original query images")
        plot_digits(query_images, query_labels)
        print("Applied the elastic transformation to the query images")
        plot_digits(query_images_def[-20:], support_labels_def[-20:])

    assert support_images_def.shape == (no_def, 1, 28, 28), \
        f"Shape of support_images_def is not {no_def}x1x28x28, but {support_images_def.shape}"
    assert query_images_def.shape == (no_def, 1, 28, 28), \
        f"Shape of query_images_def is not {no_def}x1x28x28, but {query_images_def.shape}"
    assert support_labels_def.shape == (no_def, 1), \
        f"Shape of support_labels_def is not {no_def}x1, but {support_labels_def.shape}"
    assert query_labels_def.shape == (no_def, 1), \
        f"Shape of query_labels_def is not {no_def}x1, but {query_labels_def.shape}"
    
    support_images_aug[no_rot + no_trn + no_sld:no_rot + no_trn + no_sld + no_def] = support_images_def
    support_labels_aug[no_rot + no_trn + no_sld:no_rot + no_trn + no_sld + no_def] = support_labels_def
    query_images_aug[no_rot + no_trn + no_sld:no_rot + no_trn + no_sld + no_def] = query_images_def
    query_labels_aug[no_rot + no_trn + no_sld:no_rot + no_trn + no_sld + no_def] = query_labels_def

    if debug:
        print("query_labels_def dtype: ", support_images_def.dtype)
        print("query_labels_def min value: ", support_images_def.min().item())
        print("query_labels_def max value: ", support_images_def.max().item())
        print("query_labels_def shape: ", support_images_def.shape)

    # ^ Noising transformations with labels
    support_images_nsy, support_labels_nsy = get_noisy_transformations(support_images, support_labels)
    query_images_nsy, query_labels_nsy = get_noisy_transformations(query_images, query_labels)
    support_images_nsy = support_images_nsy.view(-1, 1, 28, 28)
    query_images_nsy = query_images_nsy.view(-1, 1, 28, 28)

    if debug:
        print("Original support images")
        plot_digits(support_images, support_labels)
        print("Applied the noising transformation to the support images")
        plot_digits(support_images_nsy[-20:], support_labels_nsy[-20:])
        print("Original query images")
        plot_digits(query_images, query_labels)
        print("Applied the noising transformation to the query images")
        plot_digits(query_images_nsy[-20:], support_labels_nsy[-20:])

    assert support_images_nsy.shape == (no_nsy, 1, 28, 28), \
        f"Shape of support_images_nsy is not {no_nsy}x1x28x28, but {support_images_nsy.shape}"
    assert query_images_nsy.shape == (no_nsy, 1, 28, 28), \
        f"Shape of query_images_nsy is not {no_nsy}x1x28x28, but {query_images_nsy.shape}"
    assert support_labels_nsy.shape == (no_nsy, 1), \
        f"Shape of support_labels_nsy is not {no_nsy}x1, but {support_labels_nsy.shape}"
    assert query_labels_nsy.shape == (no_nsy, 1), \
        f"Shape of query_labels_nsy is not {no_nsy}x1, but {query_labels_nsy.shape}"
    
    support_images_aug[no_rot + no_trn + no_sld + no_def:] = support_images_nsy
    support_labels_aug[no_rot + no_trn + no_sld + no_def:] = support_labels_nsy
    query_images_aug[no_rot + no_trn + no_sld + no_def:] = query_images_nsy
    query_labels_aug[no_rot + no_trn + no_sld + no_def:] = query_labels_nsy

    if debug:
        print("support_images_nsy dtype: ", support_images_nsy.dtype)
        print("support_images_nsy min value: ", support_images_nsy.min().item())
        print("support_images_nsy max value: ", support_images_nsy.max().item())
        print("support_images_nsy shape: ", support_images_nsy.shape)

    return support_images_aug, support_labels_aug, query_images_aug, query_labels_aug, no_aug


# * Returns the augmented support and query dataloader from the given data to be used during training.
def get_augmented_support_query_loader(support_images_aug, support_labels_aug, query_images_aug, query_labels_aug, 
                query_batch_size, debug=False):
    
    support_dataset = TensorDataset(support_images_aug, support_labels_aug)
    support_loader = DataLoader(support_dataset, batch_size=query_batch_size, shuffle=True)
    query_dataset = TensorDataset(query_images_aug, query_labels_aug)
    query_loader = DataLoader(query_dataset, batch_size=query_batch_size, shuffle=True)

    if debug:
        print("*********************************")
        print("CHECK AUGMENTED SUPPORT SET")
        indices = torch.randperm(len(support_images_aug))[:50]
        plot_digits(support_images_aug[indices], support_labels_aug[indices])

        print("*********************************")
        print("CHECK AUGMENTED QUERY SET")
        query_images_sample = []
        query_labels_sample = []
        for i, data in enumerate(query_loader):
            image, labels = data
            query_images_sample.append(image)
            query_labels_sample.append(labels)
        query_images_sample = torch.cat(query_images_sample)
        query_labels_sample = torch.cat(query_labels_sample)
        indices = torch.randperm(len(query_images_sample))[:50]
        plot_digits(query_images_sample[indices], query_labels_sample[indices])
        print("*********************************")

    return support_loader, query_loader