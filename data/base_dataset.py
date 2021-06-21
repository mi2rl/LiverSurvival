import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    
    def __init__(self, all_path, label, df, input_shape, transforms_):
        self.all_path = all_path
        self.label = label
        self.df = df
        self.input_shape = input_shape
        self.transform = transforms_
        
    def __len__(self):
        return len(self.all_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data_path = self.all_path[idx]
        label = np.array(self.label[idx])
        
        img = np.load(data_path)[0]
        img = self.transform(**{'data':img[None][None]})

        sample = {'data':img['data'][0], 'path':data_path, 'label':label}

        return sample