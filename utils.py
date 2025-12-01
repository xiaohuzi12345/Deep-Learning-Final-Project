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
np.random.seed(42)

def load_label(root):
    file_path = f"{root}/data/4/train.csv"
    y = pd.read_csv(file_path, index_col=0)
    y = y.sort_values(by='file_name')
    y.loc[:, 'file_name'] = y['file_name'].apply(lambda x: x.split("/")[-1])
    y = y.set_index('file_name')
    return y
    
def to_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def load_batch_x(image_files):
    image_vectors = []
    for file_name in tqdm(image_files):
        image_path = os.path.join(image_folder, file_name)
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img_resized) / 255.
        image_vectors.append(img_array)
    return torch.from_numpy(np.array(image_vectors).astype(np.float32))
    
def load_batch_y(image_files):
    return torch.from_numpy(y.loc[image_files].values.astype(np.float32))

def compute_accuracy(logits, labels):
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).float().mean().item()
    return correct