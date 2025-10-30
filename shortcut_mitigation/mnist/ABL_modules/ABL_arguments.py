from argparse import Namespace

# * args for loading the supervised dataset
args_add = Namespace(
    dataset='addmnist',     
    batch_size=32,
    preprocess=0,
    c_sup=1,
    which_c=[-1],
    model='mnistsl',        
    task='addition',    
)

# * args for loading the unsupervised dataset
args_short = Namespace(
    dataset='shortmnist',     
    batch_size=32,
    preprocess=0,
    c_sup=0,
    which_c=[-1],
    model='mnistsl',  # dummy model      
    task='addition',
    lr=0.001,
    weight_decay=0.0001, 
)