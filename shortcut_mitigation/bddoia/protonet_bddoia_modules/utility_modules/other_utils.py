import torch
import pickle
from torch.utils.data import Dataset
from protonet_STOP_bddoia_modules.proto_modules.proto_helpers import get_random_classes


# * method used to check if all encoder parameters are registered in the model's optimizer
def check_optimizer_params(model):
    """Check that all encoder parameters are registered in the optimizer."""
    # Get all encoder parameters
    encoder_params = []
    for i in range(21):
        encoder = model.encoder[i]
        for name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen params
            encoder_params.append((f"encoder_{i}.{name}", param))

    # Get all parameters in the optimizer
    opt_param_ids = set(id(p) for group in model.opt.param_groups for p in group['params'])

    # Check each encoder param is in the optimizer
    missing = [(name, p.shape) for name, p in encoder_params if id(p) not in opt_param_ids]

    if missing:
        print("⚠️ The following parameters are missing from the optimizer:")
        for name, shape in missing:
            print(f"  - {name}: {shape}")
        raise RuntimeError("Some encoder parameters are not registered in the optimizer.")
    else:
        print("✅ All encoder parameters are correctly registered in the optimizer.")



# * semi-deterministic variant of get_random_classes where the positive examples are always the same for a given class index
def get_per_class_support_set(proto_datasets:Dataset, pos_examples:dict, class_idx:int, device:str, debug=False):
    pos_list = pos_examples[class_idx]  
    support_embeddings_pos = torch.stack([ ex['images_embeddings_raw'].unsqueeze(0) for ex in pos_list ], dim=0).to(device)
    support_labels_pos = torch.ones(len(pos_list), dtype=torch.long, device=device)
    num_pos_labels = support_labels_pos.sum().item()
    if debug:
        print(f"Class {class_idx}: {support_labels_pos.shape} embeddings, {support_labels_pos.shape} labels (all 1)")

    proto_labels = proto_datasets[class_idx].labels
    proto_data = proto_datasets[class_idx].embeddings
    
    mask = proto_labels == 0
    proto_data_neg = proto_data[mask]
    proto_labels_neg = proto_labels[mask]
    support_embeddings_neg, support_labels_neg = get_random_classes(
        proto_data_neg, proto_labels_neg, n_support=num_pos_labels, n_classes=1
    )
    if debug:
        print("Support embeddings shape: ", support_embeddings_neg.shape)
        print("Support labels shape: ", support_labels_neg.shape)
    
    assert torch.all(support_labels_neg == 0), "support_labels contains non-zero entries"

    support_embeddings_combined = torch.cat([support_embeddings_pos, support_embeddings_neg], dim=0)
    support_labels_combined = torch.cat([support_labels_pos, support_labels_neg], dim=0)
    if debug:
        print("Combined support embeddings shape:", support_embeddings_combined.shape)
        print("Combined support labels shape:", support_labels_combined.shape)

    return support_embeddings_combined, support_labels_combined


# * Save prototypes and dataloaders to a file
def save_prototypes(proto_datasets: dict, proto_dataloaders: dict, support_emb_dict: dict, seed: int):
    # Serialize proto_datasets
    dataset_serialized = {k: v.to_dict() for k, v in proto_datasets.items()}
    with open(f"proto_datasets_{seed}.pkl", "wb") as f:
        pickle.dump(dataset_serialized, f)

    # Save only useful DataLoader config options
    dataloader_configs = {
        k: {
            'batch_size': loader.batch_size,
            'shuffle': getattr(loader, 'shuffle', False),  # may not exist
            'num_workers': loader.num_workers,
            'drop_last': loader.drop_last,
            'pin_memory': loader.pin_memory
        }
        for k, loader in proto_dataloaders.items()
    }
    with open(f"proto_dataloaders_config_{seed}.pkl", "wb") as f:
        pickle.dump(dataloader_configs, f)

    # Move embeddings to CPU for portability
    cpu_embeddings = {
        k: (v[0].cpu(), v[1].cpu())
        for k, v in support_emb_dict.items()
    }

    torch.save(cpu_embeddings, f'embeddings_dict_{seed}.pt')
    print("Prototypes and dataloaders saved successfully.")