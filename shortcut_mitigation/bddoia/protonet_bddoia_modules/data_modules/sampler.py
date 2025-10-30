import torch
import numpy as np
from torch.utils.data import Dataset

class ProtoDataset(Dataset):
    def __init__(self, embeddings, labels):
        assert embeddings.shape[0] == labels.shape[0]
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
    def to_dict(self):
        return {
            'data': self.embeddings,
            'labels': self.labels,
        }

    @classmethod
    def from_dict(cls, config):
        return cls(data=config['embeddings'], labels=config['labels'])


class PrototypicalBatchSampler(object):
    """
    Yields a batch of indices for episodic training.
    At each iteration, it randomly selects 'classes_per_it' classes and then picks
    'num_samples' samples for each selected class.
    """
    def __init__(self, labels, classes_per_it, num_samples, iterations):
        """
        Args:
            labels (array-like): 1D array or list of labels for the target task.
                                 This should be either the shape labels or the colour labels.
            classes_per_it (int): Number of random classes for each iteration.
            num_samples (int): Number of samples per class (support + query) in each episode.
            iterations (int): Number of iterations (episodes) per epoch.
        """
        self.labels = np.array(labels)
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # Create an index matrix of shape (num_classes, max_samples_in_class)
        max_count = max(self.counts)
        self.indexes = np.empty((len(self.classes), max_count), dtype=int)
        self.indexes.fill(-1)
        self.indexes = torch.LongTensor(self.indexes)
        self.numel_per_class = torch.zeros(len(self.classes), dtype=torch.long)

        # Fill in the matrix with indices for each class.
        for idx, label in enumerate(self.labels):
            # Find the row corresponding to this label
            class_idx = (self.classes == label).nonzero(as_tuple=False).item()
            # Find the next available column (where the value is -1)
            pos = (self.indexes[class_idx] == -1).nonzero(as_tuple=False)[0].item()
            self.indexes[class_idx, pos] = idx
            self.numel_per_class[class_idx] += 1

    def __iter__(self):
        """
        Yield a batch of indices for each episode.
        """
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for _ in range(self.iterations):
            batch = torch.LongTensor(cpi * spc)
            # Randomly choose 'classes_per_it' classes
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, class_idx in enumerate(c_idxs):
                s = slice(i * spc, (i + 1) * spc)

                n_avail = self.numel_per_class[class_idx]
                if spc <= n_avail:
                    # enough examples → sample without replacement
                    perm = torch.randperm(n_avail)
                    sample_idxs = perm[:spc]
                else:
                    # too few → sample with replacement
                    sample_idxs = torch.randint(0, n_avail, (spc,), dtype=torch.long)

                batch[s] = self.indexes[class_idx, sample_idxs]

            # Shuffle the batch indices
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.iterations