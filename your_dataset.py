# Save this file in data_provider/your_dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class YourTimeSeriesDataset(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='your_data.csv',
                 target='AI1-01', scale=True, timeenc=0, freq='s', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        
        if size is None:
            self.seq_len = 24 * 4 * 4  # Default
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Set type
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Rename the time column to 'date' for consistency
        df_raw.rename(columns={'time[s]': 'date'}, inplace=True)
        
        # For time-based splits, calculate appropriate boundaries
        # For small datasets, you might want to adjust these ratios
        total_length = len(df_raw)
        train_length = int(total_length * 0.7)
        test_length = int(total_length * 0.2)
        val_length = total_length - train_length - test_length
        
        border1s = [0, train_length - self.seq_len, train_length + val_length - self.seq_len]
        border2s = [train_length, train_length + val_length, total_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Select features
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # All columns except date
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        # Scale data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Time encoding - for your case it might be simpler to just use the timestamp as is
        df_stamp = df_raw[['date']][border1:border2]
        
        # For very frequent data like yours, you might want to create custom time features
        # or just use the raw timestamps
        if self.timeenc == 0:
            # Use timestamp directly as a feature
            time_ind = np.arange(len(df_stamp))
            data_stamp = np.expand_dims(time_ind, axis=1)
        elif self.timeenc == 1:
            # For high-frequency data, you might need to adjust time_features
            # or create your own time encoding
             time_ind = np.arange(len(df_stamp))
             data_stamp = np.expand_dims(time_ind, axis=1)  # Make it 2D: [time_steps, 1]
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
    
    # Ensure we don't exceed the dataset bounds
        if r_end > len(self.data_x):
            return self.__getitem__(index - 1)
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
    
    # Create proper time features - reshape to ensure 2D format
        if len(seq_x_mark.shape) == 1:  # If it's 1D
            seq_x_mark = np.expand_dims(seq_x_mark, axis=1)
        if len(seq_y_mark.shape) == 1:  # If it's 1D
            seq_y_mark = np.expand_dims(seq_y_mark, axis=1)
    
    # Ensure that time features are 2D, not 3D or 4D
        if len(seq_x_mark.shape) > 2:
            seq_x_mark = seq_x_mark.reshape(seq_x_mark.shape[0], -1)
        if len(seq_y_mark.shape) > 2:
            seq_y_mark = seq_y_mark.reshape(seq_y_mark.shape[0], -1)
    
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)