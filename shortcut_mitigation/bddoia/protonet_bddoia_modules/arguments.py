import torch
from argparse import Namespace


# * Arguments for DPL
args_dpl = Namespace(
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
    dataset='boia_original_embedded', # ^ processed dataset
    entity='', 
    entropy=True,
    exp_decay=0.9,                  
    gamma=1e-3,                      
    imp_op='Prod',
    joint=False, 
    lr=0.001,                       
    model='probddoiadpl',      
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
    task='boia', # ^ do not change 
    tuning=False, 
    use_ood=False, 
    val_metric='accuracy', 
    validate=False, 
    w_c=1, 
    w_h=1,                          
    w_rec=1, 
    w_sl=10,
    w_kl=1,
    wandb=None, 
    warmup_steps=0, 
    weight_decay=0.0001,    # * bddoia            
    which_c=[-1],
    device=torch.device("cuda"),

    # & prototypes related arguments
    prototypes=True,             
    n_support=75,
    embedding_dim=64,
    debug=False,
    proto_epochs=1,
    patience=15,
    hide = [],
    boia_stop=False,
    expressive_model=False,

    num_support = 5,                         # ✅ Episodic training
    num_query = 5,                           # ✅ Episodic training
    num_samples = 10,                        # ✅ Episodic training (num_support + num_query
    classes_per_it = 2,                      # ✅ Episodic training
    iterations = 100,                        # ✅ Episodic training
)