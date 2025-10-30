import torch
from argparse import Namespace

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
    dataset='shortmnist',         
    entity='', 
    entropy=False, 
    exp_decay=0.9, 
    gamma=1e-3,                   
    imp_op='Prod',                
    joint=False, 
    lr=0.001,                     
    model='promnistsl',  # ^ add 'pro' in front of a model name to run its prototypical network version
    n_epochs=40,
    non_verbose=False, 
    notes=None, 
    or_op='Prod',                 
    p=2,                          
    posthoc=False, 
    preprocess=False, 
    proj_name='', 
    project='Reasoning-Shortcuts', 
    seeds=[], 
    splitted=False, 
    task='addition', 
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
    weight_decay=1e-4,            
    which_c=[-1],
    device=torch.device("cuda"),

    # & prototypes related arguments
    prototypes=True,             
    prototypical_loss_weight=[1.0],
    prototypical_dataset='addmnist', # ^ dataset with complete digits to create support and query set
    prototypical_batch_size=32,
    n_support=75,                            # 75
    embedding_dim=64,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    patience=5,
    hide=[],

    num_support=5,                         # 5
    num_query=5,                           # 5
    num_samples=10,                        # 10
    classes_per_it=5,                      # ✅ Episodic training
    iterations=100,                        # ✅ Episodic training

    no_interaction=False,  
    mlp=True                         
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
    dataset='shortmnist',      
    entity='', 
    entropy=True,
    exp_decay=0.9, 
    gamma=1e-3,
    imp_op='Prod',
    joint=False, 
    lr=0.001, 
    model='promnistltn', 
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
    task='addition', 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=10, 
    w_rec=1, 
    w_sl=10, 
    wandb=None, 
    warmup_steps=0, 
    weight_decay=0.0001,
    which_c=[-1],
    device=torch.device("cuda"),
    
    # & prototypes related arguments
    prototypes=True,             
    prototypical_loss_weight=[1.0],
    prototypical_dataset='addmnist', # ^ dataset with complete digits to create support and query set
    prototypical_batch_size=32,
    n_support=2,                             # 75
    embedding_dim=64,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    patience=5,
    hide = [],

    num_support = 1,                         # 5
    num_query = 1,                           # 5
    num_samples = 2,                         # 10
    classes_per_it = 5,                      # ✅ Episodic training
    iterations = 100,                        # ✅ Episodic training
    
    no_interaction=False,
)


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
    dataset='shortmnist',           
    entity='', 
    entropy=True,             
    exp_decay=0.9,                  
    gamma=1e-3,                      
    imp_op='Prod',
    joint=False, 
    lr=0.001,                       
    model='promnistdpl',      
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
    task='addition', 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=1,                          
    w_rec=0, 
    w_sl=10,
    wandb=None, 
    warmup_steps=0, 
    weight_decay=0.0001,            
    which_c=[-1],
    device=torch.device("cuda"),

    # & prototypes related arguments
    prototypes=True,             
    prototypical_loss_weight=[1.0],
    prototypical_dataset='addmnist', # ^ dataset with complete digits to create support and query set
    prototypical_batch_size=32,
    n_support=2,                             # 75
    embedding_dim=64,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    patience=5,
    hide = [],

    num_support = 1,                         # 5
    num_query = 1,                           # 5
    num_samples = 2,                         # 10
    classes_per_it = 5,                      # ✅ Episodic training
    iterations = 100,                        # ✅ Episodic training
    
    no_interaction=False,
)

# * Arguments for CCN+
args_ccn = Namespace(
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
    dataset='shortmnist',         
    entity='', 
    entropy=False, 
    exp_decay=0.9, 
    gamma=1e-3,                   
    imp_op='Prod',                
    joint=False, 
    lr=0.001,                     
    model='shieldedmnist',  # ^ add 'pro' in front of a model name to run its prototypical network version
    n_epochs=40,
    non_verbose=False, 
    notes=None, 
    or_op='Prod',                 
    p=2,                          
    posthoc=False, 
    preprocess=False, 
    proj_name='', 
    project='Reasoning-Shortcuts', 
    seeds=[], 
    splitted=False, 
    task='addition', 
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
    weight_decay=1e-4,            
    which_c=[-1],
    device=torch.device("cuda"),

    # & prototypes related arguments
    prototypes=False,             
    prototypical_loss_weight=[1.0],
    prototypical_dataset='addmnist', # ^ dataset with complete digits to create support and query set
    prototypical_batch_size=32,
    n_support=75,                             # 75
    embedding_dim=64,
    debug=False,
    proto_lr=0.001,
    proto_epochs=10,
    patience=5,
    hide = [],

    num_support = 5,                         # 5
    num_query = 5,                           # 5
    num_samples = 10,                        # 10
    classes_per_it = 5,                      # ✅ Episodic training
    iterations = 100,                        # ✅ Episodic training
    
    no_interaction=False,
)