import matplotlib.pyplot as plt
import torch


# * plots concept images with the corresponding labels (shape and colour)
def plot_primitives(images, labels, num_images=50):

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