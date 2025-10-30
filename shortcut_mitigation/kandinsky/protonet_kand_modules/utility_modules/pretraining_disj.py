import os 
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))

from utils import fprint
from utils.status import progress_bar


def evaluate_model(model, data_loader):
    model.encoder.eval()
    correct_shape = 0
    correct_color = 0
    total = 0

    with torch.no_grad():
        for images, shape_labels, color_labels in data_loader:
            images = images.to(model.device)
            shape_labels = shape_labels.to(model.device)
            color_labels = color_labels.to(model.device)

            # Forward through separate backbones
            shape_preds = model.encoder[0](images)
            color_preds = model.encoder[1](images)

            shape_pred_labels = torch.argmax(shape_preds, dim=1)
            color_pred_labels = torch.argmax(color_preds, dim=1)

            correct_shape += (shape_pred_labels == shape_labels).sum().item()
            correct_color += (color_pred_labels == color_labels).sum().item()
            total += images.size(0)

    shape_acc = correct_shape / total
    color_acc = correct_color / total
    overall_acc = (shape_acc + color_acc) / 2

    print(f"Shape Accuracy: {shape_acc:.4f}")
    print(f"Color Accuracy: {color_acc:.4f}")
    print(f"Overall Accuracy: {overall_acc:.4f}")


def pre_train(model, train_loader, args, seed: int = 0):
    
    # Full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False

    # Separate optimizers
    enc_opt = torch.optim.Adam(
        list(model.encoder[0].parameters()) + list(model.encoder[1].parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    fprint("\n--- Start of Training ---\n")
    model.encoder.to(model.device)
    model.encoder.train()

    shape_loss_fn = torch.nn.CrossEntropyLoss()
    color_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.proto_epochs):
        for i, batch in enumerate(train_loader):
            images, shape_labels, color_labels = batch
            images = images.to(model.device)
            shape_labels = shape_labels.to(model.device)
            color_labels = color_labels.to(model.device)
            batch_size = images.size(0)

            assert images.shape == torch.Size([batch_size, 3, 64, 64]), \
                f"Expected shape [{batch_size}, 3, 64, 64], but got {images.shape}"

            enc_opt.zero_grad()

            # Forward pass through separate backbones
            shape_preds = model.encoder[0](images)
            color_preds = model.encoder[1](images)

            assert shape_preds.shape == (batch_size, 3), \
                f"Expected shape_preds ({batch_size}, 3), but got {shape_preds.shape}"
            assert color_preds.shape == (batch_size, 3), \
                f"Expected color_preds ({batch_size}, 3), but got {color_preds.shape}"

            # Compute losses
            loss_shape = shape_loss_fn(shape_preds, shape_labels)
            loss_color = color_loss_fn(color_preds, color_labels)
            loss = loss_shape + loss_color

            loss.backward()
            enc_opt.step()

            progress_bar(i, len(train_loader), epoch, loss.item())

    evaluate_model(model, train_loader)