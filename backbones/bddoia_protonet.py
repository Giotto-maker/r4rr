import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import Module


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


class ProtoNetConv1D(nn.Module):
    # ! add some in_dim logic to make the model size agnostic and use the same architecture for both scene ad object identification
    def __init__(self, in_dim=3072, z_dim=256):
        """
        Initializes the encoder network for the model.

        Args:
            in_dim (int, optional): Input feature dimension. Default is 3072.
            z_dim (int, optional): Output embedding dimension. Default is 256.

        The encoder consists of a sequence of 1D convolutional, batch normalization, activation, pooling, and linear layers.
        It processes input tensors of shape [B, 1, in_dim] and outputs embeddings of dimension z_dim.
        """
        super().__init__()
        self.in_dim = in_dim
        self.encoder = nn.Sequential(
            # reshape will be done outside: [B, 1, 3072]
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # [B, 32, 1536]

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # [B, 64, 768]

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [B, 128, 1]

            nn.Flatten(),  # [B, 128]
            nn.Linear(128, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU()
        )


    def forward(self, x):
        x = x.view(x.size(0), 1, self.in_dim)  
        return self.encoder(x)
    

    def predict(self,
                support_features: torch.Tensor,  # [n_support, in_dim]
                support_labels:   torch.Tensor,  # [n_support]
                query_features:   torch.Tensor,  # [n_query,   in_dim]
                debug: bool=False):
        """
        Few‑shot predict on 1D “images” (feature vectors).

        Args:
          support_features: Tensor [n_support, in_dim]
          support_labels:   Tensor [n_support]  ints in {0,1}
          query_features:   Tensor [n_query, in_dim]
          debug:            bool

        Returns:
          y_hat:   Tensor [n_query]         predicted class indices  
          log_p_y: Tensor [n_query, C]      log‑probs for each class  
          logits:  Tensor [n_query, C]      raw scores = –EuclideanDist
        """
        device = next(self.encoder.parameters()).device
        support_features = support_features.to(device)
        support_labels   = support_labels.to(device)
        query_features   = query_features.to(device)

        # 1) embeddings
        support_emb = self.forward(support_features)  # [n_support, z_dim]
        query_emb   = self.forward(query_features)    # [n_query,   z_dim]

        # 2) prototypes (sorted by label to keep consistent)
        classes = torch.unique(support_labels)
        classes, _ = torch.sort(classes)
        protos = []
        for c in classes:
            idxs = (support_labels == c).nonzero(as_tuple=True)[0]
            protos.append(support_emb[idxs].mean(0))
        prototypes = torch.stack(protos)  # [C, z_dim]

        # 3) distances & log‑probs
        dists   = euclidean_dist(query_emb, prototypes)      # [n_query, C]
        log_p_y = F.log_softmax(-dists, dim=1)               # [n_query, C]
        _, y_hat = log_p_y.max(1)                            # [n_query]

        if debug:
            print("classes:", classes.tolist())
            print("distances:", dists)

        return y_hat, log_p_y, -dists
    

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

        return loss_val,  acc_val