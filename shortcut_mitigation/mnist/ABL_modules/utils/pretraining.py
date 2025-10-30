import os
import sys
import torch 
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(".."))       
sys.path.append(os.path.abspath("../.."))    

from utils import fprint
from utils.status import progress_bar


def pre_train(model, train_loader, seed: int = 0):
    
    # for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # start optimizer
    enc_opt = torch.optim.Adam(model.parameters(), 0.001, weight_decay=0.0001)

    fprint(f"\n--- Start of PreTraining with seed {seed} ---\n")
    model.to(device)

    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            sup_images = images.to(device)               # shape: (batch_size, C, 28, 28)
            sup_labels = labels.to(device)               # shape: (batch_size, 1)
            batch_size = sup_images.size(0)

            assert sup_images.shape == torch.Size([batch_size, 1, 28, 28]), \
            f"Expected shape [{batch_size}, 1, 28, 28], but got {sup_images.shape}"
            assert sup_labels.shape == torch.Size([batch_size, ]), \
            f"Expected shape [{batch_size},], but got {sup_labels.shape}"

            enc_opt.zero_grad()
            preds = model(sup_images).squeeze(1)
            assert preds.shape == (batch_size, 10),\
                f"Expected preds shape ({batch_size}, 10), but got {preds.shape}"
            
            loss = F.cross_entropy(preds, sup_labels)
            loss.backward()
            enc_opt.step()

            progress_bar(i, len(train_loader), epoch, loss.item())

    # Compute training accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).squeeze(1)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Training accuracy: {acc:.4f}")