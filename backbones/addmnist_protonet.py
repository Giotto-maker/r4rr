import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st

from torch.nn.modules import Module


# * Utils methods: convolutional block definition (OK)
def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


# * Utils methods: euclidean distance computation (OK)
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


# * Utility method to compute the variance of the prototypes
def compute_variance(prototypes: torch.Tensor, emb_dim:int) -> float:
    
    if prototypes.size(0) < 2:
        return 1.0
    centroid = prototypes.mean(dim=0).unsqueeze(0)
    dists = euclidean_dist(centroid, prototypes)
    max_distance = dists.max().item()
    chi2_quantile = st.chi2.ppf(0.99, df=emb_dim)
    variance = max_distance / chi2_quantile
    if variance > 1.0:
        print(f"[NOTE] Variance is above the 99th percentile and it is equal to: {variance}.")

    return variance


# * Utility method to initialize unknown prototypes with Gaussian noise (OK)
def initialize_unknown_prototypes_with_noise(known_prototypes, num_unknowns, variance):
    """
    Initialize unknown prototypes around the mean of known prototypes by adding
    Gaussian noise with specified variance.

    Args:
        known_prototypes (torch.Tensor): Tensor of shape [num_known_classes, emb_dim].
        num_unknowns (int): Number of unknown prototypes.
        variance (float): Variance of the Gaussian noise to add.

    Returns:
        torch.Tensor: Unknown prototypes of shape [num_unknowns, emb_dim].
    """
    emb_dim = known_prototypes.shape[1]
    device = known_prototypes.device
    centroid = known_prototypes.mean(dim=0)
    std = variance ** 0.5
    noise = torch.randn(num_unknowns, emb_dim, device=device) * std
    unknown_prototypes = centroid.unsqueeze(0) + noise
    print("[|INIT|]: Unknown prototypes initialized in the hyperball.")

    return unknown_prototypes
    


# ! Prototypical Network Class
class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    # & Using the convolutional blocks to build the prototypical network (OK)
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64, missing_classes=None):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        
        self.missing_classes = missing_classes if missing_classes is not None else []
        self.num_hidden = len(self.missing_classes)
        if self.num_hidden > 0:
            self.unknown_prototypes = nn.Parameter(torch.randn(self.num_hidden, z_dim, requires_grad=True))
        else:
            self.unknown_prototypes = None

        
    # & Compute the embeddings for support and query points to use during episodes (OK)
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
    # & Predicts the unsupervised data points (OK)
    def predict(self, support_images, support_labels, query_images, unknown_init=None, debug=False):
        """
        Given a model, a support set (support_images, support_labels) and query_images,
        computes the prototypes and returns predicted labels for the query set.
        
        Args:
            model: the ProtoNet model.
            support_images: Tensor of shape [n_support_total, C, H, W].
            support_labels: Tensor of shape [n_support_total], with labels as integers.
            query_images: Tensor of shape [n_query_total, C, H, W].
            
        Returns:
            y_hat: Tensor of shape [n_query_total] with predicted class indices.
            log_p_y: Tensor of shape [n_query_total, n_classes] with log-probabilities.
            logits: Negative distances (before softmax), as a measure of similarity.
        """
        device = next(self.encoder.parameters()).device  # get device from model

        # Ensure support and query images are on the same device as the model
        support_images = support_images.to(device)
        query_images = query_images.to(device)

        # Compute embeddings for support and query images.
        support_embeddings = torch.empty(0)
        if support_images.size(0) != 0:
            support_embeddings = self.forward(support_images)  # shape: [n_support_total, emb_dim]
        query_embeddings = self.forward(query_images)      # shape: [n_query_total, emb_dim]
        n_query = query_embeddings.size(0)
        emb_dim = query_embeddings.size(1)

        # Determine which classes are present in the support set.
        # For MNIST, assume full label set is 0-9.
        full_class_set = list(range(10))
        available = {}  # Map from label to computed prototype.

        # Compute prototypes: for each unique label in the support set, average the embeddings.
        unique_labels = torch.unique(support_labels)
        for label in unique_labels:
            idx = (support_labels == label).nonzero(as_tuple=True)[0]
            proto = support_embeddings[idx].mean(dim=0)
            available[label.item()] = proto
        
        # Sanity check: known prototypes.
        for label, proto in available.items():
            assert isinstance(proto, torch.Tensor), f"Prototype for label {label} is not a tensor."
            assert proto.size() == (emb_dim,), f"Prototype for label {label} does not have the correct size: expected ({emb_dim},), got {proto.size()}."
        assert len(available) == unique_labels.size(0), f"Number of unique prototypes {len(available)} does not match number of unique labels {unique_labels.size(0)}."

        # ^ Setting unknown prototypes
        if self.num_hidden > 0 and unknown_init:
            # ! Compute the unknown prototypes!
            known_prototypes = torch.stack(list(available.values()))
            if debug:  print("[DEBUG] Initializing unknown prototypes")
            assert known_prototypes.size(0) + self.num_hidden == 10, \
                f"Number of prototypes should be 10, but got {known_prototypes.size(0) + self.num_hidden}"
            variance = compute_variance(known_prototypes, emb_dim=emb_dim)
            unknown_prototypes = initialize_unknown_prototypes_with_noise(known_prototypes, self.num_hidden, variance)    
            with torch.no_grad():
                self.unknown_prototypes.data.copy_(unknown_prototypes)
                assert self.unknown_prototypes.shape == (self.num_hidden, emb_dim),\
                    f"Unknown prototypes shape mismatch: expected ({self.num_hidden}, {emb_dim}), got {self.unknown_prototypes.shape}"
                if debug: 
                    print("[DEBUG] Unknown prototypes initialized with first value: ", self.unknown_prototypes[0])
        
        if self.num_hidden > 0 and not unknown_init and debug:
            print("[DEBUG] Unknown prototype first value: ", self.unknown_prototypes[0])

        # Build the final list of prototypes ordered by full_class_set.
        prototypes = []
        missing_sorted = sorted(self.missing_classes)
        for c in full_class_set:
            if c in available:
                prototypes.append(available[c])
            else:
                # Find the index in the sorted missing list.
                # This index must correspond to the correct row in unknown_prototypes.
                missing_idx = missing_sorted.index(c)
                prototypes.append(self.unknown_prototypes[missing_idx])

        prototypes = torch.stack(prototypes)  # [10, emb_dim]
        assert prototypes.shape == (10, emb_dim), f"Concatenated prototypes shape mismatch: expected (10, {emb_dim}), got {prototypes.shape}"
        
        # Compute Euclidean distances between each query and each prototype.
        dists = euclidean_dist(query_embeddings, prototypes)
        assert dists.shape == (n_query, 10), f"Dists shape is not ({n_query},10), but {dists.shape}"
        
        # Apply log-softmax to the negative distances to obtain log-probabilities for each class.
        log_p_y = F.log_softmax(-dists, dim=1)
        assert log_p_y.shape == (n_query, 10), f"log_p_y shape is not ({n_query},10), but {log_p_y.shape}"
        
        # Compute predictions: for each query, choose the class with highest log-probability.
        _, y_hat = log_p_y.max(1)  # Predicted classes; y_hat is 1-dimensional.
        assert y_hat.ndim == 1, "y_hat must be a 1-dimensional tensor."
        
        # Return predictions, log-probabilities, and logits (negative distances)
        return y_hat, log_p_y, -dists
        

# ! Prototypical Loss Class
class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    # & Initialize the loss function with the number of support samples (OK)
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    # & Compute the prototypical loss function (OK)
    def forward(self, input, target):
        return self.prototypical_loss(input, target, self.n_support)

    # & Function from the original paper to compute the prototypical loss (OK)
    def prototypical_loss(self, input, target, n_support):
        '''
        Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

        Compute the barycentres by averaging the features of n_support
        samples for each class in target, computes then the distances from each
        samples' features to each one of the barycentres, computes the
        log_probability for each n_query samples for each one of the current
        classes, of appartaining to a class c, loss and accuracy are then computed
        and returned
        Args:
        - input: the model output for a batch of samples
        - target: ground truth for the above batch of samples
        - n_support: number of samples to keep in account when computing
        barycentres, for each one of the current classes
        '''
        target_cpu = target.to('cpu')
        input_cpu = input.to('cpu')

        def supp_idxs(c):
            # FIXME when torch will support where as np
            return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target_cpu)
        n_classes = len(classes)
        # FIXME when torch will support where as np
        # assuming n_query, n_target constants
        n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

        support_idxs = list(map(supp_idxs, classes))

        prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
        # FIXME when torch will support where as np
        query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

        query_samples = input.to('cpu')[query_idxs]
        dists = euclidean_dist(query_samples, prototypes)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val, acc_val