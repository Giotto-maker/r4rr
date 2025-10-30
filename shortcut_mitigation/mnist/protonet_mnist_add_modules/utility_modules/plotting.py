import matplotlib.pyplot as plt

from collections import Counter


# * Plots single digits
def plot_digits(imgs, labels, args=None):
    if args is not None and args.no_interaction: return
    assert imgs.shape[0] == labels.shape[0], "Number of images and labels must be the same"
    imgs = imgs.to('cpu')
    batch_size = imgs.shape[0]
    
    # Determine the grid dimensions
    max_cols = 10  # Maximum number of images per row
    rows = (batch_size + max_cols - 1) // max_cols  # Calculate the number of rows needed
    cols = min(batch_size, max_cols)  # Number of columns is the smaller of batch size or max_cols

    plt.figure(figsize=(cols * 2, rows * 2))  # Adjust the figure size based on rows and cols

    for i in range(batch_size):
        # Subplot indexing: row-major order
        plt.subplot(rows, cols, i + 1)
        plt.title(f"Label: {labels[i].item()}")
        plt.imshow(imgs[i].squeeze(), cmap='gray')  # Use squeeze to remove single-dimensional entries from the shape
        plt.axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()


# * Plots pair of digits for training data inspection
def plot_training_image(images, labels, args, plot_index_start=0, plot_index_end=10):
    if args.no_interaction: return
    for plotting_index in range(plot_index_start, plot_index_end + 1):
        image = images[plotting_index].cpu().numpy().transpose(1, 2, 0)
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.title(f"Label {labels[plotting_index]}")
        plt.axis('off')
        plt.show()


# * Plots an episode of the episodic dataloader
def plot_episodic_dataloader(episodic_dataloader, num_support, num_query, classes_per_it, args):
    if classes_per_it > 0:
        for batch in episodic_dataloader:
            images, labels = batch
            labels = labels.tolist()
            label_counts = Counter(labels)  # Count occurrences of each label
            
            print("Batch images shape:", images.shape)  # Should be [batch_size, 1, 28, 28] for MNIST
            print("Batch labels:", labels)  # Check if labels are grouped correctly per class
            print("Label distribution in batch:", label_counts)
            break  # Only inspect one batch

        # Plot the first 10 images
        if args.no_interaction: return
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        if len(args.hide) > 0:
            min_images = min(20, num_support + num_query)
        else:
            min_images = 20
        for i in range(min_images):
            ax = axes[i // 10, i % 10]
            ax.imshow(images[i][0], cmap="gray")  # MNIST is grayscale
            ax.set_title(f"Label: {labels[i]}")
            ax.axis("off")

        plt.show()