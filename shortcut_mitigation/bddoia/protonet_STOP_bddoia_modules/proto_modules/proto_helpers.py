import torch




def compute_prototypical_logit_per_class(y_hat, neg_dists):
    """
    Computes the prototypical logit per class based on the given predictions and distances.
    This function determines whether a concept occurs (positive class) or not (negative class)
    and calculates the confidence score accordingly. If the concept occurs, the highest confidence
    score for its occurrence is returned. Otherwise, the lowest confidence score for its absence
    is returned.
    Args:
        y_hat (torch.Tensor): A tensor of predicted labels, where each element indicates the 
                              predicted class (e.g., 1 for positive, 0 for negative).
        log_p_y (torch.Tensor): A tensor of log probabilities for each class (not used in this function).
        neg_dists (torch.Tensor): A tensor of distances or confidence scores, where the first column 
                                  corresponds to the confidence for the negative class and the second 
                                  column corresponds to the confidence for the positive class.
    Returns:
        dict: A dictionary containing:
            - "found_positive" (bool): True if the positive class is found, False otherwise.
            - "confidence" (float): The confidence score for the classification outcome. This is the 
                                    highest confidence for the positive class if found, or the lowest 
                                    confidence for the negative class otherwise.
    """
    # does the concept occur?
    pos_idxs = (y_hat == 1).nonzero(as_tuple=True)[0]
    classification_outcome = None

    # if it does, take the *highest* confidence score for its occurrence:
    if pos_idxs.numel() > 0:
        pos_confidences = neg_dists[pos_idxs, 1]    # Tensor [M]
        best_confidence = pos_confidences.max().item()
        classification_outcome = {
            "found_positive": True,
            "confidence":     best_confidence
        }
    # if it does not, take the *lowest* confidence score for its absence:
    else:
        neg_confidences = neg_dists[:, 0]           # Tensor [B]
        worst_confidence = neg_confidences.min().item()
        classification_outcome = {
            "found_positive": False,
            "confidence":     worst_confidence
        }

    return classification_outcome


def compute_class_logits_per_batch(inputs, this_support_embeddings, this_support_labels, traffic_lights_model, debug=False):
    batch_logits = []
    for idx, ci in enumerate(inputs):
        if debug:   print(f"Image {idx}: {ci.shape}  (should be Ni × 3072)")
        if ci.shape[0] == 0:
            batch_logits.append(torch.tensor(0.0))
        else:
            ci = ci.unsqueeze(1) # add channel dimension
            y_hat, _ , dists = traffic_lights_model.predict(this_support_embeddings, this_support_labels, ci)
            classification_outcome = compute_prototypical_logit_per_class(y_hat, dists)
            distances_min, distances_max = dists.min().item(), dists.max().item()
            logit_score = (classification_outcome['confidence'] - distances_min) / (distances_max - distances_min + 1e-8)
            if debug:   
                print("Classification outcome:", classification_outcome['found_positive'])
                print("Logit score:", logit_score)
            batch_logits.append(torch.tensor(logit_score))
    
    batch_logits = torch.stack(batch_logits)
    if debug:   print(len(inputs), "processed images in batch")
    return batch_logits


def get_random_classes(embeddings, labels, n_support, n_classes=2):
    """
    Selects a random subset of samples from the given embeddings and labels, ensuring that
    the specified number of classes and support samples per class are included.
    Args:
        embeddings (torch.Tensor): A tensor containing the embeddings of the samples.
        labels (torch.Tensor): A tensor containing the labels corresponding to the embeddings.
        n_support (int): The number of samples to select per class.
        n_classes (int, optional): The number of unique classes to include. Default is 2.
    Returns:
        tuple: A tuple containing:
            - selected_images (torch.Tensor): A tensor of selected embeddings.
            - selected_labels (torch.Tensor): A tensor of selected labels.
    Raises:
        AssertionError: If the number of unique classes in `labels` is not equal to `n_classes`.
        AssertionError: If there are not enough samples in a class to meet the `n_support` requirement.
    """
    unique_classes = torch.unique(labels)
    assert len(unique_classes) == n_classes, f"There should be exactly {n_classes} unique classes."

    selected_embeddings = []
    selected_labels = []

    for cls in unique_classes:
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        assert len(class_indices) >= n_support, f"Not enough samples for class {cls}"
        random_indices = torch.randperm(len(class_indices))[:n_support]
        selected_embeddings.append(embeddings[class_indices[random_indices]])
        selected_labels.append(labels[class_indices[random_indices]])

    selected_embeddings = torch.cat(selected_embeddings)
    selected_labels = torch.cat(selected_labels)

    return selected_embeddings, selected_labels


def assert_inputs(images_embeddings, attr_labels, class_labels,
                   detected_rois_feats, detected_rois, detection_labels,
                   detection_scores, images_embeddings_raw):
    """
    Asserts the shapes of the inputs to ensure they are as expected.
    Args:
        images_embeddings (torch.Tensor): Tensor of image embeddings.
        attr_labels (torch.Tensor): Tensor of attribute labels.
        class_labels (torch.Tensor): Tensor of class labels.
        detected_rois_feats (list): List of detected ROIs features.
        detected_rois (list): List of detected ROIs.
        detection_labels (list): List of detection labels.
        detection_scores (list): List of detection scores.
        images_embeddings_raw (torch.Tensor): Tensor of raw image embeddings.
    """
    batch_size = images_embeddings.shape[0]
    assert images_embeddings.shape == (batch_size, 2048), f"Expected shape [2048], but got {images_embeddings.shape}" # ok
    assert images_embeddings_raw.shape == (batch_size, 2048), f"Expected shape [2048], but got {images_embeddings_raw.shape}" # ok
    assert attr_labels.shape == (batch_size, 21), f"Expected shape [21], but got {attr_labels.shape}" # ok
    assert class_labels.shape == (batch_size, 4), f"Expected shape [5], but got {class_labels.shape}" # ok
    assert all(roi_feats.size(-1) == 1024 for roi_feats in detected_rois_feats),\
        f"Expected last dimension to be 1024, but got {[roi_feats.shape for roi_feats in detected_rois_feats]}" # ok
    assert len(detected_rois) == batch_size, f"Expected {batch_size} rois, but got {len(detected_rois)}" # ok
    assert len(detected_rois_feats) == batch_size, f"Expected {batch_size} roi_feats, but got {len(detected_rois_feats)}"  # ok
    assert len(detection_labels) == batch_size, f"Expected {batch_size} detection_labels, but got {len(detection_labels)}" # ok
    assert len(detection_scores) == batch_size, f"Expected {batch_size} detection_scores, but got {len(detection_scores)}" # ok