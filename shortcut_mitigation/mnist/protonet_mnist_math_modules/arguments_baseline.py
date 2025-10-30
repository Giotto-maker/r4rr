import torch
from argparse import Namespace

args_dpl_base = Namespace(
    and_op='Godel',
    backbone='conceptizer', 
    batch_size=64, # & ok          
    beta=0.99,                      
    boia_model='ce', 
    boia_ood_knowledge=False, 
    c_sup=0.0, 
    c_sup_ltn=0, 
    checkin=None, 
    checkout=False, 
    count=30, 
    dataset='mnmath',     
    entity='', 
    entropy=False, 
    exp_decay=0.9,                  
    gamma=1e-3,                      
    imp_op='Prod',
    joint=False, 
    lr=0.001,  # & ok                     
    model='mnmathdpl',   # ^ 'mnmathdpl' for DPL model and 'promnmathdpl' for DPL + PNet                
    n_epochs=40,                    
    non_verbose=False, 
    notes=None, 
    or_op='Prod',           
    p=6,                    
    posthoc=False, 
    preprocess=False, 
    proj_name='', 
    project='Reasoning-Shortcuts', 
    seeds=[], 
    splitted=False, 
    task='mnmath', 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=0,                       
    w_rec=1, 
    w_sl=10,
    wandb=None, 
    warmup_steps=0, 
    weight_decay=0.0001,  # & ok          
    which_c=[-1],
    device=torch.device("cuda"),
    
    seed = 1415,        # 1415, 1617, 1819, || 2021, 2223.
    patience = 15,
    prototypes=False,
    prototypical_dataset='addmnist', # ^ dataset with complete digits to create support and query set
    prototypical_batch_size=32,
    n_support=75,
    embedding_dim=64,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    hide=[],

    num_support = 5,                         # ✅ Episodic training
    num_query = 5,                           # ✅ Episodic training
    num_samples = 10,                        # ✅ Episodic training (num_support + num_query
    classes_per_it = 5,                      # ✅ Episodic training
    iterations = 100,                        # ✅ Episodic training

    no_interaction = False,
)


args_cbm_base = Namespace(
    and_op='Godel',
    backbone='conceptizer', 
    batch_size=64, # & ok          
    beta=0.99,                      
    boia_model='ce', 
    boia_ood_knowledge=False, 
    c_sup=0.05, 
    c_sup_ltn=0, 
    checkin=None, 
    checkout=False, 
    count=30, 
    dataset='mnmath',     
    entity='', 
    entropy=False, 
    exp_decay=0.9,                  
    gamma=1e-3,                      
    imp_op='Prod',
    joint=False, 
    lr=0.001,  # & ok                     
    model='mnmathcbm',   # ^ 'mnmathcbm' for CBM model and 'promnmathdpl' for DPL + PNet          
    n_epochs=40,
    non_verbose=False, 
    notes=None, 
    or_op='Prod',           
    p=6,                    
    posthoc=False, 
    preprocess=False, 
    proj_name='', 
    project='Reasoning-Shortcuts', 
    seeds=[], 
    splitted=False, 
    task='mnmath', 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=0,                       
    w_rec=1, 
    w_sl=10,
    wandb=None, 
    warmup_steps=0, 
    weight_decay=0.0001,  # & ok          
    which_c=[9],
    device=torch.device("cuda"),
    
    seed = 1415,        # 1415, 1617, 1819, || 2021, 2223.
    patience = 15,
    prototypes=False,
    prototypical_dataset='addmnist', # ^ dataset with complete digits to create support and query set
    prototypical_batch_size=32,
    n_support=75,
    embedding_dim=64,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,

    num_support = 5,                         # ✅ Episodic training
    num_query = 5,                           # ✅ Episodic training
    num_samples = 10,                        # ✅ Episodic training (num_support + num_query
    classes_per_it = 5,                      # ✅ Episodic training
    iterations = 100,                        # ✅ Episodic training

    no_interaction = False,
)