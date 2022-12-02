#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

#%%
def score(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """ 
    y_true - pandas.DataFrame: alert_key | sar_flag
    y_pred - pandas.DataFrame: alert_key | prob  
    """
    n = len(y_true[y_true.sar_flag==1]) # Num sar
    sar = y_true[y_true.sar_flag==1].alert_key.values # sar alert keys
    hit = 0
    for i, y_pred_ in enumerate(y_pred.alert_key.values):
        if y_pred_ in sar:
            hit += 1
        if hit == n-1:
            break
    return (n-1)/(i+1), n-1, i

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
# %%
label = pd.read_csv('E:/Datasets/Fintech/TrainingDataset_first/train_y_answer.csv')
pred = pd.DataFrame([], columns=['alert_key', 'probability'])
pred['alert_key'] = label.alert_key
pred['probability'] = 1
# %%
score(label, pred)
# %%
