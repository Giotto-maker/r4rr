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

    fprint("\n--- Start of Pre-Training ---\n")
    model.encoder.to(model.device)
    model.encoder.train()

    for epoch in range(args.proto_epochs):
        for i, (images, labels) in enumerate(train_loader):
            sup_images = images.to(model.device)               
            sup_labels = labels.to(model.device)
            batch_size = sup_images.size(0)

            assert sup_images.shape == torch.Size([batch_size, 1, 28, 28]), \
            f"Expected shape [{batch_size}, 1, 28, 28], but got {sup_images.shape}"
            assert sup_labels.shape == torch.Size([batch_size, ]), \
            f"Expected shape [{batch_size},], but got {sup_labels.shape}"

            enc_opt.zero_grad()
            preds = model.encoder(sup_images)[0].squeeze(1)
            assert preds.shape == (batch_size, 10), f"Expected preds shape ({batch_size}, 10), but got {preds.shape}"
            
            loss = F.cross_entropy(preds, sup_labels)
            loss.backward()
            enc_opt.step()

            progress_bar(i, len(train_loader), epoch, loss.item())

    # Compute training accuracy
    model.encoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            preds = model.encoder(images)[0].squeeze(1)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Training accuracy: {acc:.4f}")