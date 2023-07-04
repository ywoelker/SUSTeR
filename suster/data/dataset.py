import pickle
import torch 
from torch.utils.data import Dataset

class DatasetMetr(Dataset):

    def __init__(self, datafile, indexfile, mode, percent = 1.0) -> None:
        super().__init__()

        with open(datafile, 'rb') as dataf:
            data = pickle.load(dataf)

        with open(indexfile, 'rb') as indf:
            self.index = pickle.load(indf)

        processed_data = data["processed_data"]
        processed_drop_data = data["processed_drop_data"]
        self.data = torch.from_numpy(processed_data).float()
        self.drop_data = torch.from_numpy(processed_drop_data).float()
        
        self.index = self.index[mode]

        if percent < 1.0:
            self.index = self.index[: int(len(self.index) * percent)]

        self.scaler_mean = torch.tensor(data['scaler_values'][0])
        self.scaler_scale = torch.tensor(data['scaler_values'][1])

    def __getitem__(self, index):
        idx = list(self.index[index])
        history_data = self.drop_data[idx[0]:idx[1]]
        future_data = self.data[idx[1]:idx[2]]
        return history_data, future_data
    
    def __len__(self):
        return len(self.index)