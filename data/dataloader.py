import numpy as np
import torch
from torch.utils.data import Dataset

class Liver_CustomDataset_surv(Dataset):
    def __init__(self, all_path, df, input_shape, transforms_, breaks, img=True):
        """Initialize the dataset with paths, dataframe, and other configurations."""
        self.all_path = all_path
        self.df = df
        self.input_shape = input_shape
        self.transform = transforms_
        self.breaks = breaks
        self.img = img
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.all_path)

    def extract_tx(self, idn):
        """Extract treatment data from the dataframe for a given id."""
        cols = [25, 26, 24, 29, 27, 31]  # Define columns to extract
        data = self.df.loc[self.df['id'] == int(idn), self.df.columns[cols]].values.flatten()
        return torch.tensor(data, dtype=torch.long)

    def normalize(self, p_id, category):
        """Normalize a dataframe column value for a given patient id."""
        value = self.df.loc[self.df['id'] == p_id, category].item()
        min_val = self.df[category].min()
        max_val = self.df[category].max()
        return (value - min_val) / (max_val - min_val)
        
    def pad(self, img, cval=0):
        """Apply padding to the image to match the target input shape."""
        padding = [(int(np.ceil((t - s) / 2)), int(np.floor((t - s) / 2))) for s, t in zip(img.shape, self.input_shape)]
        return np.pad(img, padding, mode='constant', constant_values=cval)

    def center_crop(self, img):
        """Crop the center of the image to match the target input shape."""
        slices = [slice(max((s - t) // 2, 0), min((s + t) // 2, s)) for s, t in zip(img.shape, self.input_shape)]
        return img[tuple(slices)]

    def make_surv_array(self, time, flag, breaks):
        """Generate a survival array based on the breaks and patient data."""
        n_intervals = len(breaks) - 1
        midpoints = breaks[:-1] + np.diff(breaks) / 2
        y_train = np.zeros(n_intervals * 2)
        target_breaks = breaks[1:] if flag else midpoints
        y_train[:n_intervals] = 1.0 * (time >= target_breaks)
        if time < breaks[-1]:
            y_train[n_intervals + np.where(time < breaks[1:])[0][0]] = 1
        return y_train

    def __getitem__(self, idx):
        """Retrieve a single item from the dataset."""
        data_path = self.all_path[idx if not isinstance(idx, torch.Tensor) else idx.item()]

        if self.img:
            all_data = np.load(data_path)
            img_data = self.pad(self.center_crop(all_data[0]))
            img_data = self.transform(data=img_data[None, None])
        else:
            img_data = {'data': [1]}

        idn = int(data_path.split('/')[-1].split('_')[1])
        patient_data = self.df[self.df['id'] == idn]
        surv, mon = patient_data['death_01'].item(), patient_data['death_mo'].item()

        sample = {
            'data': img_data['data'][0], 'path': data_path,
            'surv_f': torch.tensor(self.make_surv_array(mon, surv, self.breaks)[:len(self.breaks) // 2]),
            'surv_s': torch.tensor(self.make_surv_array(mon, surv, self.breaks)[len(self.breaks) // 2:]),
            'death': surv, 'death_mo': mon, 'tx': self.extract_tx(idn),
            'feature': torch.tensor([self.normalize(idn, cat) for cat in [
                'age', 'BMI', 'ECOG', 'Child', 'varix', 'ascites', 'AFP', 'Hb', 'PLT',
                'ALT', 'TB', 'alb', 'PT_INR', 'Cr']])
        }
        return sample
