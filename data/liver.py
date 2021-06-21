import pandas as pd
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import random
from utils import random_seed_, tr_val_test
from data.base_dataset import BaseDataset
from torch.utils.data import DataLoader

def load_dataloader(df_path, data_path, tr_transforms, val_transforms, tr_bs, val_bs, n_cpu):
    random_seed_(opt.random_seed)

    # Data Load
    df=pd.read_excel(df_path)

    path_list = subfiles(data_path, suffix='.npy')
    idx_list = [f.split('_')[1] for f in subfiles(data_path, join=False, suffix='.npy')]

    cls0_path_list=[]
    cls1_path_list=[]

    for idx, path in zip(idx_list, path_list):
        sample_df = df[df['id']==int(idx)]

        if (sample_df['death_01'].item()==1) & (sample_df['death_mo'].item()<=14):
            cls0_path_list.append(path)
        elif (sample_df['death_01'].item()==1):
            cls1_path_list.append(path)

    random.shuffle(cls0_path_list)
    random.shuffle(cls1_path_list)


    cls0_tr, cls0_val, cls0_test = tr_val_test(cls0_path_list)
    cls1_tr, cls1_val, cls1_test = tr_val_test(cls1_path_list)

    tr_data_list = np.concatenate((cls0_tr, cls1_tr))
    val_data_list = np.concatenate((cls0_val, cls1_val))
    test_data_list = np.concatenate((cls0_test, cls1_test))

    tr_label_list = [0]*len(cls0_tr) + [1]*len(cls1_tr)
    val_label_list = [0]*len(cls0_val) + [1]*len(cls1_val)
    test_label_list = [0]*len(cls0_test) + [1]*len(cls1_test)


    train_dataset = BaseDataset(tr_data_list, tr_label_list, df, opt.patch_size, tr_transforms)
    val_dataset = BaseDataset(val_data_list, val_label_list, df, opt.patch_size, val_transforms)
    test_dataset = BaseDataset(test_data_list, test_label_list, df, opt.patch_size, val_transforms)

    dataloader = DataLoader(train_dataset, batch_size=tr_bs, shuffle=True, num_workers=n_cpu)
    val_dataloader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return dataloader, val_dataloader, test_dataloader