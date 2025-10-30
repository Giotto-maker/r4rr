import torch

from collections import Counter
from torch.utils.data import DataLoader
from protonet_bddoia_modules.data_modules.sampler import ProtoDataset, PrototypicalBatchSampler


# * This function builds prototypical dataloaders for few-shot learning using positive and negative examples per class.
def build_prototypical_dataloaders(unsup_train_loader, device, args, debug=True):
    """
    Builds prototypical dataloaders for few-shot learning using positive and negative examples per class.
    This function processes an unsupervised training data loader to extract positive and negative examples for each class,
    constructs per-class datasets, and returns PyTorch DataLoaders with prototypical batch sampling for each class.
    Args:
        unsup_train_loader (DataLoader): PyTorch DataLoader yielding batches with keys:
            - 'embeddings_raw': list of raw image embeddings (torch.Tensor)
            - 'rois': list of region of interest tensors per image
            - 'roi_feats': list of ROI feature tensors per image
            - 'detection_labels': detection label tensors per image
            - 'detection_scores': detection score tensors per image
            - 'attr_labels': attribute label tensors per image (multi-label, shape [B, 21])
        device (torch.device): Device to move tensors to (e.g., 'cuda' or 'cpu').
        args (Namespace): Arguments namespace with attributes:
            - classes_per_it (int): Number of classes per iteration (episode).
            - num_samples (int): Number of samples per class per batch.
            - iterations (int): Number of iterations (batches) per epoch.
            - num_support (int): Number of support examples per class in each batch.
            - num_query (int): Number of query examples per class in each batch.
        debug (bool, optional): If True, prints debug information for the first example. Default is True.
    Returns:
        pos_examples (dict): Dictionary mapping class index (int) to list of positive example dicts.
        proto_datasets (dict): Dictionary mapping class index (int) to ProtoDataset objects containing:
            - 'embeddings': torch.Tensor of shape [N, 1, 2048]
            - 'labels': torch.Tensor of shape [N], with 1 for positive and 0 for negative examples
        proto_dataloaders (dict): Dictionary mapping class index (int) to DataLoader objects using
            PrototypicalBatchSampler for episodic sampling.
    Notes:
        - The function ensures that positive and negative sets are disjoint for each class.
        - Multi-label augmentation is performed: examples with multiple positive labels are added to all relevant classes.
        - Negative examples for a class are those that are positive for other classes but not for the current class.
        - Extensive sanity checks are performed to ensure data integrity and correct label assignment.
    """
    # Init positive examples (21 classes, each with list of examples) and number of positives per class
    pos_examples = {cls_idx: [] for cls_idx in range(21)}
    target_per_class = 6
    
    # Loop over dataset until we collect target_per_class for each class
    for batch_idx, batch in enumerate(unsup_train_loader):
        raw_embs = torch.stack(batch['embeddings_raw']).to(device)
        rois = batch['rois']
        roi_feats = batch['roi_feats']
        det_labels = batch['detection_labels']
        det_scores = batch['detection_scores']
        attrs = torch.stack(batch['attr_labels']).to(device)  # shape [B,21]
        batch_size = attrs.size(0)

        for b in range(batch_size):
            attr_vector = attrs[b].clone().cpu()
            for cls in torch.nonzero(attr_vector).flatten().tolist():
                if len(pos_examples[cls]) >= target_per_class:
                    continue
                example = {
                    'source_id': (batch_idx, b),
                    'images_embeddings_raw': raw_embs[b].detach().cpu().clone(),
                    'detected_rois': [r.clone() for r in rois[b]],
                    'detected_rois_feats': [f.detach().cpu().clone() for f in roi_feats[b]],
                    'detection_labels': det_labels[b].detach().cpu().clone(),
                    'detection_scores': det_scores[b].detach().cpu().clone(),
                    'attr_labels': attr_vector,
                    'is_positive': True
                }
                if debug:
                    for key, value in example.items():
                        if torch.is_tensor(value):
                            print(f"{key}: {value.shape}")
                        elif isinstance(value, list) and len(value) and torch.is_tensor(value[0]):
                            print(f"{key}: list of {len(value)} tensors, first shape: {value[0].shape}")
                        else:
                            print(f"{key}: {type(value)}")
                    debug = False
                pos_examples[cls].append(example)

        # Check if all classes reached target
        if all(len(pos_examples[c]) >= target_per_class for c in range(21)):
            break

    # Rebuild neg_examples after any augmentation to keep sets disjoint
    neg_examples = {cls_idx: [] for cls_idx in range(21)}

    # First, allow multi-label augmentation: add any example with attr_labels[i]==1 to pos_examples[i] if it is new
    for cls in range(21):
        seen_ids = {ex['source_id'] for ex in pos_examples[cls]}
        for other_cls in range(21):
            if other_cls == cls:
                continue
            for ex in pos_examples[other_cls]:
                if ex['attr_labels'][cls] == 1 and ex['source_id'] not in seen_ids:
                    new_ex = ex.copy()
                    new_ex['is_positive'] = True
                    pos_examples[cls].append(new_ex)
                    seen_ids.add(ex['source_id'])

    # Now build negatives: any example that has attr_labels[i]==0 but appears in any pos_examples of other classes
    for cls in range(21):
        seen_ids_pos = {ex['source_id'] for ex in pos_examples[cls]}
        for other_cls in range(21):
            if other_cls == cls:
                continue
            for ex in pos_examples[other_cls]:
                if ex['attr_labels'][cls] == 0 and ex['source_id'] not in seen_ids_pos:
                    neg_ex = ex.copy()
                    neg_ex['is_positive'] = False
                    neg_examples[cls].append(neg_ex)

    # Ensure no overlap between pos and neg
    for cls in range(21):
        assert not set(ex['source_id'] for ex in neg_examples[cls]) & set(ex['source_id'] for ex in pos_examples[cls]), \
            f"Overlap in pos/neg for class {cls}"
        
    # Initialize dataset per class
    dataset_per_class = {}
    for cls in range(21):
        examples = pos_examples[cls] + neg_examples[cls]
        emb_list, label_list = [], []
        for ex in examples:
            emb_list.append(ex['images_embeddings_raw'].unsqueeze(0))
            label_list.append(1 if ex['is_positive'] else 0)
        embeddings_tensor = torch.stack(emb_list).to(device)  # [N,1,2048]
        labels_tensor = torch.tensor(label_list, device=device)
        dataset_per_class[cls] = {'embeddings': embeddings_tensor, 'labels': labels_tensor}

    '''
     dataset_per_class contains, for each class i:
        (a) 'embeddings': tensor([N,1,2048])
        (b) 'labels': tensor([N]) with values 1 for positive, 0 for negative
    '''

    for cls in range(21):
        print(f"Class {cls}: embeddings shape = {dataset_per_class[cls]['embeddings'].shape}, labels shape = {dataset_per_class[cls]['labels'].shape}")
        emb = dataset_per_class[cls]['embeddings']
        lab = dataset_per_class[cls]['labels']
        
        # Exact count assertion and label values are correct check
        expected_total = len(pos_examples[cls]) + len(neg_examples[cls])
        assert emb.size(0) == expected_total, f"Class {cls} count mismatch: {emb.size(0)} vs {expected_total}"
        assert set(lab.tolist()) <= {0,1}, f"Invalid labels for class {cls}"
        pos_count = len(pos_examples[cls])
        neg_count = len(neg_examples[cls])
        # positives should be labeled 1 in the first pos_count entries
        for idx in range(pos_count):
            assert lab[idx].item() == 1, f"Positive at wrong pos for class {cls}, idx {idx}"
        # negatives should be labeled 0 in the next neg_count entries
        for idx in range(neg_count):
            assert lab[pos_count + idx].item() == 0, f"Negative at wrong pos for class {cls}, idx {pos_count + idx}"

    print("Dataset per class built with explicit flags and no overlaps.")

    proto_datasets = {}
    proto_dataloaders = {}

    for cls in range(21):
        proto_data = dataset_per_class[cls]['embeddings']
        proto_labels = dataset_per_class[cls]['labels']
        proto_datasets[cls] = ProtoDataset(proto_data, proto_labels)
        proto_sampler = PrototypicalBatchSampler(
                        labels = proto_labels.cpu().numpy(),
                        classes_per_it = args.classes_per_it,
                        num_samples = args.num_samples,
                        iterations = args.iterations,
                    )
        proto_dataloaders[cls] = DataLoader(proto_datasets[cls], batch_sampler=proto_sampler)

    # Labels count check for proto_datasets[cls] and proto_dataloaders[cls]
    for cls in range(21):
        # Dataset label count
        label_counter_dataset = Counter(proto_datasets[cls].labels.cpu().tolist())
        print(f"Class {cls} - Dataset Label 0 count: {label_counter_dataset[0]}, Label 1 count: {label_counter_dataset[1]}")

        # Dataloader label count
        label_counter_loader = Counter()
        for batch in proto_dataloaders[cls]:
            _, labels = batch
            label_counter_loader.update(labels.tolist())
        print(f"Class {cls} - Dataloader Label 0 count: {label_counter_loader[0]}, Label 1 count: {label_counter_loader[1]}")

    # Final sanity Check
    for cls in range(21):
        print(f"Class {cls}: Dataset size = {len(proto_datasets[cls])}, Dataloader batches = {len(proto_dataloaders[cls])}")
        assert len(proto_dataloaders[cls]) == args.iterations, \
            f"Class {cls}: Expected {args.iterations} batches, got {len(proto_dataloaders[cls])}"
        for batch in proto_dataloaders[cls]:
            embeddings, labels = batch
            #print("Batch Embeddings Shape:", embeddings.shape, "Labels Shape:", labes.shape)
            assert embeddings.shape == ((args.num_support + args.num_query) * args.classes_per_it, 1, 2048), \
                f"Embeddings shape mismatch: {embeddings.shape}"
            assert labels.shape == ((args.num_support + args.num_query) * args.classes_per_it,), \
                f"Labels shape mismatch: {labels.shape}"
            
    return pos_examples, proto_datasets, proto_dataloaders