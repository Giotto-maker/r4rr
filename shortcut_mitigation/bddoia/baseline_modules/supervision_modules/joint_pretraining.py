import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))  

from utils import fprint
from utils.status import progress_bar


def pre_train(model, train_loader, args, seed: int = 0):

    # for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False

    # start optimizer
    enc_opt = torch.optim.Adam(model.encoder.parameters(), args.lr, weight_decay=args.weight_decay)

    fprint("\n--- Start of Training ---\n")
    model.encoder.to(model.device)
    
    for epoch in range(args.proto_epochs):
        for i, batch in enumerate(train_loader):
            batch_embeds, batch_labels = batch
            batch_embeds = batch_embeds.to(model.device)
            batch_labels = batch_labels.to(model.device)

            enc_opt.zero_grad()
            preds = model.encoder(batch_embeds)
            assert preds.shape == (batch_embeds.shape[0], 21), f"Expected shape ({batch_embeds.shape[0]}, 21), got {preds.shape}"
            loss = F.binary_cross_entropy(preds, batch_labels.float())

            loss.backward()
            enc_opt.step()

            progress_bar(i, len(train_loader), epoch, loss.item())