import numpy as np
import pandas as pd
import os

def score(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """ 
    y_true - pandas.DataFrame: alert_key | sar_flag
    y_pred - pandas.DataFrame: alert_key | prob  
    """
    n = len(y_true[y_true.sar_flag==1]) # Num sar
    sar = y_true[y_true.sar_flag==1].alert_key.values # sar alert keys
    y_pred.sort_values(by=['prob'], ascending=False, inplace=True)
    hit = 0
    for i, y_pred_ in enumerate(y_pred.alert_key.values):
        if y_pred_ in sar:
            hit += 1
        if hit == n-1:
            break
    return (n-1)/(i+1), n-1, i

def create_dir(path):
    while not os.path.isdir(path):
        print('path not exist')
        try:
            os.mkdir(path)
        except FileNotFoundError:
            par_dir = '/'.join(path.split('/')[:-2]) + '/'
            print(f'par_dir is {par_dir}')
            create_dir(par_dir)
        except FileExistsError:
            pass