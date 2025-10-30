import torch
from argparse import Namespace

# * Arguments for DPL
args_dpl = Namespace(
    and_op='Godel',
    backbone='conceptizer', 
    batch_size=32,          
    beta=0.99,                      
    boia_model='ce', 
    boia_ood_knowledge=False, 
    c_sup=0.0, 
    c_sup_ltn=0, 
    checkin=None, 
    checkout=False, 
    count=30, 
    dataset='kandinsky',           
    entity='', 
    entropy=True,  
    exp_decay=0.9,                  
    gamma=1e-3,                      
    imp_op='Prod',
    joint=False, 
    lr=0.0001,                       
    model='prokanddpl',                
    n_epochs=40,                    
    non_verbose=False, 
    notes=None, 
    or_op='Prod',           
    p=6,                    
    posthoc=False, 
    preprocess=False, 
    proj_name='', 
    project='Reasoning-Shortcuts', 
    seeds=[0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], 
    splitted=False, 
    task='patterns', 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=1,                          
    w_rec=1, 
    w_sl=10,
    wandb=None, 
    warmup_steps=0, 
    weight_decay=0.0001,
    which_c=[-1],
    device=torch.device("cuda"),

    # & prototypes related arguments
    prototypes=True,             
    prototypical_batch_size=32,
    n_support=75,                             # 75 aug / 3 no aug
    embedding_dim=1024,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    patience=5,
    prototypical_loss_weight=[1.0],
    extractor_training_epochs=20,
    retrain_extractor=False,
    concept_extractor_path='ultralytics/finetuned/kand_best_100.pt',
    
    # & protonet related arguments
    num_support = 5,                         # 5 aug / 1 no aug
    num_query = 5,                           # 5 aug / 2 no aug
    num_samples = 10,                        # 10 aug / 3 no aug
    classes_per_it = 3,                      # 3
    iterations = 100,                        # 100
    num_distinct_labels = 3                  # 3
)


# * Arguments for SEMANTIC LOSS
args_sl = Namespace(
    and_op='Prod',                
    backbone='conceptizer', 
    batch_size=32,                
    beta=0.99, 
    boia_model='ce', 
    boia_ood_knowledge=False, 
    c_sup=0.0,
    c_sup_ltn=0, 
    checkin=None, 
    checkout=False, 
    count=30, 
    dataset='kandinsky',         
    entity='', 
    entropy=False, 
    exp_decay=0.9, 
    gamma=1e-3,                   
    imp_op='Prod',                
    joint=False, 
    lr=0.001,                     
    model='prokandsl',
    n_epochs=40,
    non_verbose=False, 
    notes=None, 
    or_op='Prod',                 
    p=2,                          
    posthoc=False, 
    preprocess=False, 
    proj_name='', 
    project='Reasoning-Shortcuts', 
    seeds=[0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], 
    splitted=False, 
    task='patterns', 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=1,                        
    w_rec=1, 
    w_sl=10.0,                      
    wandb=None, 
    warmup_steps=0, 
    weight_decay=1e-4,            
    which_c=[-1],
    device=torch.device("cuda"),

    # & prototypes related arguments
    prototypes=True,             
    prototypical_batch_size=32,
    n_support=75,                             # 75 aug / 3 no aug
    embedding_dim=1024,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    patience=5,
    prototypical_loss_weight=[1.0],
    extractor_training_epochs=20,
    retrain_extractor=False,
    concept_extractor_path='ultralytics/finetuned/kand_best_100.pt',

    # & protonet related arguments
    num_support = 5,                         # 5 aug / 1 no aug
    num_query = 5,                           # 5 aug / 2 no aug
    num_samples = 10,                        # 10 aug / 3 no aug
    classes_per_it = 3,                      # 3
    iterations = 100,                        # 100
    num_distinct_labels = 3                  # 3
)


# * Arguments for LTN
args_ltn = Namespace(
    and_op='Godel', 
    backbone='conceptizer', 
    batch_size=64,
    beta=0.99,       
    boia_model='ce', 
    boia_ood_knowledge=False, 
    c_sup=0.0, 
    c_sup_ltn=0, 
    checkin=None, 
    checkout=False, 
    count=30, 
    dataset='kandinsky',      
    entity='', 
    entropy=False,
    exp_decay=0.9, 
    gamma=1e-3,
    imp_op='Prod',
    joint=False, 
    lr=0.001, 
    model='prokandltn', 
    n_epochs=40,
    non_verbose=False, 
    notes=None, 
    or_op='Prod',
    p=8, 
    posthoc=False, 
    preprocess=False, 
    proj_name='', 
    project='Reasoning-Shortcuts', 
    seeds=[0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], 
    splitted=False, 
    task='patterns', 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=0.8, 
    w_rec=1, 
    w_sl=10, 
    wandb=None, 
    warmup_steps=0, 
    weight_decay=0.001,
    which_c=[-1],
    device=torch.device("cuda"),

    # & prototypes related arguments
    prototypes=True,             
    prototypical_batch_size=32,
    n_support=3,                             # 75 aug / 3 no aug
    embedding_dim=1024,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    patience=5,
    prototypical_loss_weight=[1.0],
    extractor_training_epochs=20,
    retrain_extractor=False,
    concept_extractor_path='ultralytics/finetuned/kand_best_100.pt',
    
    # & protonet related arguments
    num_support = 5,                         # 5 aug / 1 no aug
    num_query = 5,                           # 5 aug / 2 no aug
    num_samples = 10,                        # 10 aug /  3 no aug
    classes_per_it = 3,                      # 3
    iterations = 100,                        # 100
    num_distinct_labels = 3                  # 3
)