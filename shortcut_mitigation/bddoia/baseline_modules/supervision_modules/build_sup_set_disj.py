import torch
from argparse import Namespace
from torch.utils.data import DataLoader
from baseline_modules.supervision_modules.sup_dataset import ProtoDataset


# * obtain augmented training DataLoader sampling positive and negative examples from the unsupervised dataset
def get_augmented_train_loader(unsup_train_loader: DataLoader, device: str, args: Namespace) -> DataLoader:

    print("----------------------------------------------------------------")
    print("\n--- Building Supervised Set for Disj Backbone Training ---\n")
    pos_examples = {cls_idx: [] for cls_idx in range(21)}
    target_per_class = 6
    debug = True

    # Loop over dataset until we collect target_per_class for each class
    for batch_idx, batch in enumerate(unsup_train_loader):
        raw_embs = torch.stack(batch['embeddings_raw']).to(device)
        attrs = torch.stack(batch['attr_labels']).to(device)  # shape [B,21]
        batch_size = attrs.size(0)

        for b in range(batch_size):
            attr_vector = attrs[b].clone().cpu()  # clone to avoid in-place issues
            for cls in torch.nonzero(attr_vector).flatten().tolist():
                if len(pos_examples[cls]) >= target_per_class:
                    continue
                example = {
                    'source_id': (batch_idx, b),
                    'images_embeddings_raw': raw_embs[b].detach().cpu().clone(),
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

    neg_examples = {cls_idx: [] for cls_idx in range(21)}

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

    for cls in range(21):
        assert not set(ex['source_id'] for ex in neg_examples[cls]) & set(ex['source_id'] for ex in pos_examples[cls]), \
            f"Overlap in pos/neg for class {cls}"
        
    dataset_per_class = {}
    for cls in range(21):
        examples = pos_examples[cls] + neg_examples[cls]
        emb_list, label_list = [], []
        for ex in examples:
            emb_list.append(ex['images_embeddings_raw'].unsqueeze(0))
            label_list.append(ex['attr_labels'])
        embeddings_tensor = torch.stack(emb_list).to(device)  # [N,1,2048]
        labels_tensor = torch.stack(label_list)
        dataset_per_class[cls] = {'embeddings': embeddings_tensor.squeeze(1), 'labels': labels_tensor}
            
    for cls in range(21):
        print(f"Class {cls}: embeddings shape = {dataset_per_class[cls]['embeddings'].shape}, labels shape = {dataset_per_class[cls]['labels'].shape}")

    supervised_dataloaders = {}
    for cls in range(21):
        supervised_data = dataset_per_class[cls]['embeddings']
        supervised_labels = dataset_per_class[cls]['labels']
        supervised_dataset = ProtoDataset(supervised_data, supervised_labels)
        supervised_dataloaders[cls] = DataLoader(
            supervised_dataset, 
            batch_size=args.batch_size,
            shuffle=True,       
        )

    print("----------------------------------------------------------------")
    return supervised_dataloaders