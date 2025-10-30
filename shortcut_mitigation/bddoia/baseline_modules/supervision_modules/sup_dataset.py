from torch.utils.data import Dataset

# * Dataset class to store augmented embeddings and labels
class ProtoDataset(Dataset):
    def __init__(self, embeddings, labels):
        assert embeddings.shape[0] == labels.shape[0]
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]