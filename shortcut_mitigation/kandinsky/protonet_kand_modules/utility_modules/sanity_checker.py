import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


def assert_my_labels(shape_labels, color_labels, kand_proto_dataset, episodic_shape_dataloader, episodic_color_dataloader):
    # Extract the 1D label arrays from the dataset labels. Note: support_dataset.labels is a tensor of shape [N,2].
    num_distinct_shape_labels = np.unique(shape_labels).size

    # Should have 3 labels for shapes (0: square, 1: circle, 2: triangle) and 3 labels for colours (0: red, 1: yellow, 2: blue)
    print(f"Number of distinct shape labels: {num_distinct_shape_labels}")
    num_distinct_color_labels = np.unique(color_labels).size
    print(f"Number of distinct color labels: {num_distinct_color_labels}")

    # Prototypical networks expects nunpy arrays for labels
    assert isinstance(shape_labels, np.ndarray), "shape labels should be a numpy.ndarray"
    assert isinstance(color_labels, np.ndarray), "color labels should be a numpy.ndarray"

    # Check tensor shapes and values
    assert kand_proto_dataset.images.shape == (shape_labels.size, 3, 64, 64), \
        "The shape of kand_proto_dataset.images should be (number of shape labels, 3, 64, 64)"
    assert kand_proto_dataset.images.shape == (color_labels.size, 3, 64, 64), \
        "The shape of kand_proto_dataset.images should be (number of color labels, 3, 64, 64)"
    assert kand_proto_dataset.labels.shape == (color_labels.size, 2), \
        "The shape of mnist_dataset.labels should be (number of shape labels, 1)"
    assert kand_proto_dataset.labels.shape == (color_labels.size, 2), \
        "The shape of mnist_dataset.labels should be (number of color labels, 1)"
    assert kand_proto_dataset.images.min() >= 0 and kand_proto_dataset.images.max() <= 1, \
        "The values of kand_proto_dataset.images should be between 0 and 1"
    assert np.all(np.isin(shape_labels, [0, 1, 2])), "Shape labels should only contain values 0, 1, or 2"
    assert np.all(np.isin(color_labels, [0, 1, 2])), "Color labels should only contain values 0, 1, or 2"

    # ^ Inspect the shape and color dataloaders
    ### Inspect one batch from the shape dataloader
    print("Shape-based episodic batch:")
    for batch in episodic_shape_dataloader:
        images, shape_labels_batch, color_labels_batch = batch
        shape_labels_list = shape_labels_batch.tolist()
        label_counts = Counter(shape_labels_list)
        print("Batch images shape:", images.shape)  # Expected: [batch_size, 3, 64, 64]
        print("Batch shape labels:", shape_labels_list)
        print("Shape label distribution in batch:", label_counts)
        break
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = images[i].permute(1, 2, 0).numpy() # Convert tensor from (3, 64, 64) to (64, 64, 3) for display
        ax.imshow(img)
        ax.set_title(f"Shape Label: {shape_labels_list[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    ### Inspect one batch from the color dataloader
    print("\nColor-based episodic batch:")
    for batch in episodic_color_dataloader:
        images, shape_labels_batch, color_labels_batch = batch
        # We only need the color labels for the color network
        color_labels_list = color_labels_batch.tolist()
        label_counts = Counter(color_labels_list)
        print("Batch images shape:", images.shape)  # Expected: [batch_size, 3, 64, 64]
        print("Batch color labels:", color_labels_list)
        print("Color label distribution in batch:", label_counts)
        break
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = images[i].permute(1, 2, 0).numpy() # Convert tensor from (3, 64, 64) to (64, 64, 3) for display
        ax.imshow(img)
        ax.set_title(f"Color Label: {color_labels_list[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()