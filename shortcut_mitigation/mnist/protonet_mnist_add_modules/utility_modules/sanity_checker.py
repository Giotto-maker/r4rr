import numpy as np

def assert_my_labels(args, support_labels_aug, mnist_dataset):
    assert all(lbl not in args.hide for lbl in mnist_dataset.labels), "mnist_dataset contains images with hidden labels"

    original_labels = support_labels_aug.squeeze().numpy()
    labels = mnist_dataset.labels.squeeze().numpy()
    num_distinct_labels = np.unique(labels).size
    print(f"Number of distinct labels: {num_distinct_labels}")

    assert labels.size == original_labels.size - mnist_dataset.num_filtered, \
        "The number of elements in labels should be the number of elements in original_labels minus the number of filtered elements"    
    assert isinstance(labels, np.ndarray), "labels should be a numpy.ndarray"
    assert mnist_dataset.images.shape == (labels.size, 1, 28, 28), \
        "The shape of mnist_dataset.images should be (number of labels, 1, 28, 28)"
    assert mnist_dataset.labels.shape == (labels.size, 1), \
        "The shape of mnist_dataset.labels should be (number of labels, 1)"
    
    if args.debug:
        print(support_labels_aug.shape)
        print(mnist_dataset.labels.shape)
    print("Classes in dataset:", np.unique(labels))
    
    