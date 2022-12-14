#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from utils import score

#%%

class seqDataset(Dataset):
    def __init__(self, seq, label):
        #print(data.shape)
        self.n_samples = len(label)
        self.seq_data = torch.from_numpy(seq)
        self.label = torch.from_numpy(label)
    def __getitem__(self, index):
        return self.seq_data[index], self.label[index]
    def __len__(self):
        return self.n_samples

#%%
label = pd.read_csv('E:/Datasets/Fintech/TrainingDataset_first/train_y_answer.csv')
data = pd.read_csv('./combined_data.csv')
data.fillna(-1, inplace=True)   
max_len = 0
for ak in tqdm(data.alert_key.unique()):
    if len(data[data.alert_key==ak]) > max_len:
        max_len = len(data[data.alert_key==ak])
        max_ak = ak
# %%
label = pd.read_csv('E:/Datasets/Fintech/TrainingDataset_first/train_y_answer.csv')
pred = pd.DataFrame([], columns=['alert_key', 'probability'])
pred['alert_key'] = label.alert_key
pred['probability'] = 1
# %%
score(label, pred)
# %%
