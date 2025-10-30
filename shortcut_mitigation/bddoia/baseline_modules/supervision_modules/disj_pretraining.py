import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))  

from utils import fprint
from utils.status import progress_bar

def pre_train(
        model, 
        dataloaders, 
        args, 
        eval_concepts: list = None,
        seed: int = 0
    ):

    # ^ for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False

    # ^ create optimizers and schedulers for each encoder
    enc_opts, enc_schs = {}, {}
    for c, name in enumerate(eval_concepts):
        opt = torch.optim.Adam(model.encoder[c].parameters())
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        enc_opts[c], enc_schs[c] = opt, sch

    # ^ move each encoder to device
    for b in range(len(model.encoder)):
        model.encoder[b].train()
        model.encoder[b].to(model.device)
    
    fprint("\n--- Start of PreTraining ---\n")
    for epoch in range(args.proto_epochs):
        
        # ^ for each concept extractor, train its encoder
        for k, name in enumerate(eval_concepts):
            dl = dataloaders[k]
            opt, sch = enc_opts[k], enc_schs[k]
            fprint(f"\n--- Pretraining of {name} ---\n")
            
            # training
            for i, batch in enumerate(dl):
                batch_embeds, batch_labels = batch
                batch_embeds = batch_embeds.to(model.device)
                batch_labels = batch_labels.to(model.device)

                opt.zero_grad()
                preds = model.encoder[k](batch_embeds)
                assert preds.shape == (batch_embeds.shape[0], 1),\
                    f"Expected shape ({batch_embeds.shape[0]}, 1), got {preds.shape}"
                
                loss = F.binary_cross_entropy(
                        preds.squeeze(1), 
                        batch_labels[:, k].float()
                    )
                loss.backward()
                opt.step()

                progress_bar(i, len(dl), epoch, loss.item())
            
            sch.step()

            # ^ evaluation
            model.encoder[k].eval()
            eval_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for batch in dl:  # if you have a validation loader, replace with that
                    batch_embeds, batch_labels = batch
                    batch_embeds = batch_embeds.to(model.device)
                    batch_labels = batch_labels.to(model.device)

                    preds = model.encoder[k](batch_embeds).squeeze(1)
                    eval_loss += F.binary_cross_entropy(
                        preds, batch_labels[:, k].float(), reduction="sum"
                    ).item()

                    pred_labels = (preds > 0.5).long()
                    correct += (pred_labels == batch_labels[:, k]).sum().item()
                    total += batch_embeds.size(0)
            
            avg_loss = eval_loss / total
            accuracy = correct / total
            fprint(f"[Epoch {epoch+1}] {name} Eval Loss: {avg_loss:.4f}, "
                   f"Accuracy: {accuracy:.4f}")
            
            model.encoder[k].train()