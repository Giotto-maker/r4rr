import os
import time
import torch

from argparse import Namespace
from torch.utils.data import DataLoader
from datasets.utils.base_dataset import BaseDataset
from datasets.utils.sddoia_creation import CONCEPTS_ORDER


from backbones.bddoia_protonet import ProtoNetConv1D
from backbones.boia_linear import BOIAConceptizer
from backbones.boia_mlp import BOIAMLP


class FasterBDDOIADataset(BaseDataset):
    NAME = "boia_original_embedded"

    """
    Dataset for FASTER-BDDOIA folder structure.
    Each sample folder contains:
        - embedded_image.pt
        - embedded_image_raw.pt
        - detected_rois.pt
        - detected_rois_feats.pt
        - detection_labels.pt
        - detection_scores.pt
        - attr_labels.pt
        - class_labels.pt
    """
    def __init__(
        self,
        args: Namespace,
        root_dir: str = 'FASTER-BDDOIA',
        split: str = 'train',
        transform=None
    ):
        """
        Args:
            args (Namespace): argument namespace containing configuration
            root_dir (str): path to FASTER-BDDOIA folder
            split (str): one of 'train', 'val', 'test'
            transform (callable, optional): optional transform on image embeddings
        """
        super().__init__(args)
        # Adjust root_dir if running from 'notebooks' directory
        cwd = os.getcwd()
        if cwd.endswith('notebooks'):
            root_dir = os.path.join('..', root_dir)
        self.split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.split_dir):
            raise ValueError(f"Split directory not found: {self.split_dir}")
        # List all sample subfolders
        self.samples = sorted(
            [os.path.join(self.split_dir, d) for d in os.listdir(self.split_dir)
             if os.path.isdir(os.path.join(self.split_dir, d))]
        )
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_dir = self.samples[idx]
        # Load tensors
        emb = torch.load(os.path.join(sample_dir, 'embedded_image.pt')).detach()
        emb_raw = torch.load(os.path.join(sample_dir, 'embedded_image_raw.pt')).detach()
        rois = torch.load(os.path.join(sample_dir, 'detected_rois.pt')).detach()
        feats = torch.load(os.path.join(sample_dir, 'detected_rois_feats.pt')).detach()
        det_labels = torch.load(os.path.join(sample_dir, 'detection_labels.pt')).detach()
        scores = torch.load(os.path.join(sample_dir, 'detection_scores.pt')).detach()
        attr_labels = torch.load(os.path.join(sample_dir, 'attr_labels.pt')).detach()
        class_labels = torch.load(os.path.join(sample_dir, 'class_labels.pt')).detach()

        # Optionally apply transform on embeddings
        if self.transform:
            emb = self.transform(emb)

        return {
            'embeddings': emb,
            'embeddings_raw': emb_raw,
            'rois': rois,
            'roi_feats': feats,
            'detection_labels': det_labels,
            'detection_scores': scores,
            'attr_labels': attr_labels,
            'class_labels': class_labels,
        }

    def get_data_loaders(
        self,
        args: Namespace,
        root_dir: str = 'FASTER-BDDOIA',
        batch_size: int = 32,
        num_workers: int = 0
    ):
        start = time.time()
        # Create datasets, passing args to each
        train_ds = FasterBDDOIADataset(args, root_dir, split='train', transform=None)
        val_ds = FasterBDDOIADataset(args, root_dir, split='val', transform=None)
        test_ds = FasterBDDOIADataset(args, root_dir, split='test', transform=None)

        print(f"Dataset sizes - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
        print(f"Loaded datasets in {time.time() - start:.2f}s")

        # Create loaders
        self.train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size if hasattr(args, 'batch_size') else batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=FasterBDDOIADataset.detection_collate,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size if hasattr(args, 'batch_size') else batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=FasterBDDOIADataset.detection_collate,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size if hasattr(args, 'batch_size') else batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=FasterBDDOIADataset.detection_collate,
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return BOIAMLP(), None
        
        elif self.args.prototypes and not self.args.boia_stop:
            print("[PROTO-INFO] Using Prototypical Networks as backbone")
            return tuple(ProtoNetConv1D(in_dim=2048) for _ in range(21)), None
        
        elif self.args.model == 'bddoiadpldisj':
            return tuple(BOIAConceptizer(din=2048, nconcept=1) for _ in range(21)), None

        return BOIAConceptizer(din=2048, nconcept=21), None

    def get_split(self):
        return 1, ()
    
    def get_concept_labels(self):
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)
        return sorted_concepts

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.train_loader))
        print("Validation samples", len(self.val_loader))
        print("Test samples", len(self.test_loader))

    def detection_collate(batch):
        """
        Custom collate_fn for batching dicts with variable-length per-sample fields.
        Each field in the returned dict is a list of tensors, length == batch_size.
        """
        collated = {}
        # assume every element of batch is a dict with identical keys
        for key in batch[0].keys():
            collated[key] = [sample[key] for sample in batch]
        return collated
