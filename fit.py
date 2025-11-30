#!/usr/bin/env python3

import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gc
import argparse
import utils
from model2 import myCNN
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_idx", default="0,1,2,3")
    parser.add_argument("--validate_idx", default="29")
    parser.add_argument("--oos_idx", default="28,30,31")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test", action="store_true", help="run in test mode")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    ROOT = "/Users/yuxin/codes/cs230"
    y = utils.load_label(ROOT)
    DIM = 256
    target_size = (DIM, DIM)
    image_tensor = []
    image_folder = f"{ROOT}/data/4/train_data"
    x_train, y_train = [], []
    for idx in args.train_idx.split(","):
        x_train.append(torch.load(f'data_batch_{idx}.pt'))
        y_train.append(torch.load(f'label_batch_{idx}.pt'))
    x_train = torch.concat(x_train, dim=0)
    y_train = torch.concat(y_train, dim=0)

    x_validate, y_validate = [], []
    for idx in args.validate_idx.split(","):
        x_validate.append(torch.load(f'data_batch_{idx}.pt'))
        y_validate.append(torch.load(f'label_batch_{idx}.pt'))
    x_validate = torch.concat(x_validate, dim=0)
    y_validate = torch.concat(y_validate, dim=0)
    if args.test:
        x_train = x_train[:2000]
        y_train = y_train[:2000]
        x_validate = x_validate[:500]
        y_validate = y_validate[:500]
    print(x_train.shape, y_train.shape)
    print(x_validate.shape, y_validate.shape)
    
    train_dataset = TensorDataset(x_train, y_train)
    validate_dataset = TensorDataset(x_validate, y_validate)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=y_validate.shape[0], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myCNN(input_channels=x_train.shape[-1]).to(device)
    if args.checkpoint is not None:
        print(f'load model {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = args.epochs
    train_acc_path = []
    train_loss_path = []
    validate_acc_path = []
    validate_loss_path = []
    X_val = []
    Y_val = []
    for xb, yb in validate_loader:    # one-time to stack; not per epoch
        X_val.append(xb)
        Y_val.append(yb)
    X_val = torch.cat(X_val).to(device)
    Y_val = torch.cat(Y_val).to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # [batch, 1]
            loss = criterion(logits, yb)  # shapes match: [batch, 1]
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_acc += utils.compute_accuracy(logits, yb) * xb.size(0)
        model.eval()
        with torch.no_grad():
            logits = model(X_val)               # [N, 1]
            probs = torch.sigmoid(logits)       # [N, 1]
            preds = (probs > 0.5).long().squeeze()    # [N]
            labels = Y_val.long().squeeze()           # [N]
            validate_acc = (preds == labels).float().mean().item()
            validate_loss = criterion(logits, Y_val).item()
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_acc / len(train_loader.dataset)
        epoch_validate_loss = validate_loss #/ len(validate_loader.dataset)
        epoch_validate_acc = validate_acc #/ len(validate_loader.dataset)
        train_acc_path.append(epoch_acc)
        train_loss_path.append(epoch_loss)
        validate_acc_path.append(epoch_validate_acc)
        validate_loss_path.append(epoch_validate_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f} Validate Loss: {epoch_validate_loss:.4f}  Validate Acc: {epoch_validate_acc:.4f}")
        results_path = f"{ROOT}/results2/{args.train_idx}" + ('.test' if args.test else "")
        checkpoints_path = f"{ROOT}/results2/{args.train_idx}" + ('.test' if args.test else "") + '/checkpoints'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)
        torch.save(train_acc_path, f"{results_path}/train_acc.pt")
        torch.save(train_loss_path, f"{results_path}/train_loss.pt")
        torch.save(validate_acc_path, f"{results_path}/validate_acc.pt")
        torch.save(validate_loss_path, f"{results_path}/validate_loss.pt")
        torch.save(model.state_dict(), f"{results_path}/model.pt")
        torch.save(train_acc_path, f"{checkpoints_path}/train_acc.{epoch}.pt")
        torch.save(train_loss_path, f"{checkpoints_path}/train_loss.{epoch}.pt")
        torch.save(validate_acc_path, f"{checkpoints_path}/validate_acc.{epoch}.pt")
        torch.save(validate_loss_path, f"{checkpoints_path}/validate_loss.{epoch}.pt")
        torch.save(model.state_dict(), f"{checkpoints_path}/model.{epoch}.pt")
