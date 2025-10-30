import os
import torch
from tqdm import tqdm


def build_filtered_concat_inputs(
    images_emb_raw: torch.Tensor,
    detected_rois:      list,
    detected_rois_feats:list,
    detection_labels:   list,
    detection_scores:   list,
    label: int,
    score: float = 0.9
) -> list:
    """
    For each image in the batch:
      1. Filter its ROIs by `label` and `score`.
      2. Tile the image's 2048-dim embedding to match the number of kept ROIs.
      3. Concat each ROI's 1024-dim feature → (Ni, 3072).

    Returns:
      List of length B, each tensor of shape (Ni, 3072). If Ni=0, shape is (0, 3072).
    """
    B, D_img = images_emb_raw.size()
    D_roi = detected_rois_feats[0].size(1)  # assume consistent ROI feature dim
    out = []

    for i in range(B):
        emb_raw = images_emb_raw[i]  # (2048,)
        rois_i   = detected_rois[i]       # (Mi, 5) or similar
        feats_i  = detected_rois_feats[i] # (Mi, 1024)
        labs_i   = detection_labels[i]    # (Mi,)
        scores_i = detection_scores[i]    # (Mi,)

        # 1. filter mask
        mask = (labs_i == label) & (scores_i >= score)
        feats_l = feats_i[mask]           # (Ni, 1024)

        Ni = feats_l.size(0)
        if Ni == 0:
            # no valid ROIs → empty (0, 3072)
            out.append(torch.empty(0, D_img + D_roi, device=emb_raw.device))
        else:
            # 2. tile image embedding: (Ni, 2048)
            emb_expand = emb_raw.unsqueeze(0).expand(Ni, -1)
            # 3. concat → (Ni, 3072)
            emb_expand = emb_expand.to(feats_l.device)  # ensure same device
            out.append(torch.cat([feats_l, emb_expand], dim=1))

    return out


def build_stop_filtered_concat_inputs(
        images_embeddings_raw: torch.Tensor,
        detected_rois: list,
        detected_rois_feats: list,
        detection_labels: list,
        detection_scores: list,
    ) -> tuple:
    """
    Builds and returns concatenated input tensors for different object categories by filtering detected regions of interest (ROIs) based on their labels and scores.
    Args:
        images_embeddings_raw (torch.Tensor): The raw image embeddings tensor.
        detected_rois (list): List of detected regions of interest (ROIs).
        detected_rois_feats (list): List of feature tensors corresponding to each detected ROI.
        detection_labels (list): List of detection labels for each ROI.
        detection_scores (list): List of detection scores for each ROI.
    Returns:
        tuple: A tuple containing concatenated input tensors for the following categories, in order:
            - traffic lights
            - traffic signs
            - cars
            - pedestrians
            - riders
            - others
    Note:
        The function internally calls `build_filtered_concat_inputs` for each category, filtering by the corresponding label and, for some categories, a minimum score.
    """

    concat_inputs_traffic_lights = build_filtered_concat_inputs(
                    images_embeddings_raw,
                    detected_rois,
                    detected_rois_feats,
                    detection_labels,
                    detection_scores,
                    label=1,    # traffic lights
                )
            
    concat_input_traffic_signs = build_filtered_concat_inputs(
                    images_embeddings_raw,
                    detected_rois,
                    detected_rois_feats,
                    detection_labels,
                    detection_scores,
                    label=2,    # traffic sign
                )
    
    concat_input_cars = build_filtered_concat_inputs(
                    images_embeddings_raw,
                    detected_rois,
                    detected_rois_feats,
                    detection_labels,
                    detection_scores,
                    label=7,    # car
                )
    
    concat_input_pedestrians = build_filtered_concat_inputs(
                    images_embeddings_raw,
                    detected_rois,
                    detected_rois_feats,
                    detection_labels,
                    detection_scores,
                    label=3,    # pedestrian
                )
    
    concat_input_riders = build_filtered_concat_inputs(
                    images_embeddings_raw,
                    detected_rois,
                    detected_rois_feats,
                    detection_labels,
                    detection_scores,
                    label=4,    # rider
                    score=0.0
                )
    
    concat_input_others = build_filtered_concat_inputs(
                    images_embeddings_raw,
                    detected_rois,
                    detected_rois_feats,
                    detection_labels,
                    detection_scores,
                    label=5,    # others
                    score=0.0
                )

    return concat_inputs_traffic_lights, \
            concat_input_traffic_signs, \
            concat_input_cars, \
            concat_input_pedestrians, \
            concat_input_riders, \
            concat_input_others


def get_prototypical_datasets_inputs(concat_inputs, input_index, attr_labels, device):
    """
    Processes a list of input tensors and their corresponding attribute labels to prepare prototypical datasets for few-shot learning.
    Args:
        concat_inputs (list of torch.Tensor): List of input tensors, each of shape [N_i, D], where N_i is the number of samples in the i-th class.
        input_index (int): Index of the attribute/column to select from the attribute labels for prototypical learning.
        attr_labels (list of torch.Tensor): List of attribute label tensors, each of shape [1, C], where C is the number of attributes (should be 21).
        device (torch.device or str): Device to which tensors should be moved (e.g., 'cpu' or 'cuda').
    Returns:
        tuple:
            - concat_inputs (torch.Tensor): Concatenated input tensors of shape [N_total, 1, D], where N_total is the total number of valid samples.
            - proto_labels (torch.Tensor): 1D tensor of selected attribute labels for each sample, shape [N_total].
            - support_size (int): The minimum count of samples with label 0 or 1 in the selected attribute column (for balanced support set construction).
    Raises:
        AssertionError: If the attribute label tensors do not have 21 columns.
        AssertionError: If the selected proto_labels tensor is not 1-dimensional.
    """
    valid_inputs = []
    valid_labels = []
    for j, tensor in enumerate(concat_inputs):
        if tensor.size(0) > 0:
            valid_inputs.append(tensor.to(device))
            valid_labels.append(attr_labels[j].repeat(tensor.size(0), 1))  # repeat label Ni times
    concat_inputs = torch.cat(valid_inputs, dim=0).unsqueeze(1)  # [N_total, 1, 3072]
    proto_labels = torch.cat(valid_labels)  # [N_valid, C]
    assert proto_labels.shape[1] == 21, f"Expected proto_labels to have 21 columns, but got {proto_labels.shape[1]}"
    
    proto_labels = proto_labels[:, input_index].to(device)  # select the third column and ensure shape is [N, 1]
    assert proto_labels.dim() == 1, f"Expected proto_labels to be a 1D tensor, but got {proto_labels.dim()}D tensor"
    num_zeros = (proto_labels == 0).sum().item()
    num_ones = (proto_labels == 1).sum().item()
    support_size = min(num_zeros, num_ones)

    #print("Valid inputs shape: ", concat_inputs.shape)
    #print("Valid labels shape: ", proto_labels.shape)
    #print()

    return concat_inputs, proto_labels, support_size


def train_my_prototypical_network(proto_loader, iterations, pNet_model, pNet_optimizer, pNet_loss):
    """
    Trains a prototypical network for a specified number of iterations using the provided data loader, model, optimizer, and loss function.
    Args:
        proto_loader (Iterable): DataLoader or iterable yielding batches of (features, labels) for prototypical episodes.
        iterations (int): Number of training iterations (episodes) to run.
        pNet_model (torch.nn.Module): The prototypical network model to be trained.
        pNet_optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        pNet_loss (callable): Loss function that computes the prototypical loss and accuracy, given embeddings and labels.
    Returns:
        tuple: Two lists containing the loss and accuracy values for each iteration (epoch_loss, epoch_acc).
    """
    epoch_loss, epoch_acc = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Device must be 'cuda' for training the prototypical network."
    for features_batch in tqdm(proto_loader, total=iterations):
        # ------------------
        # * Process prototypical episode
        # ------------------
        pNet_optimizer.zero_grad()
        input_features, input_labels = features_batch
        input_features, input_labels = input_features.to(device), input_labels.to(device)

        # Forward pass: compute embeddings for all images in the episode.
        features_embeddings = pNet_model(input_features)
        
        # Compute prototypical loss.
        features_loss, features_acc = pNet_loss(input=features_embeddings, target=input_labels)
        features_loss.backward()
        pNet_optimizer.step()

        epoch_loss.append(features_loss.item())
        epoch_acc.append(features_acc.item())

    return epoch_loss, epoch_acc