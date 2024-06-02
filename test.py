import argparse
import torch
import torch.nn as nn
from augment import tr_transforms, val_transforms
import SimpleITK as sitk
import pandas as pd
import numpy as np
from time import time, sleep
from torch.utils.data import DataLoader
from dataloader import Liver_CustomDataset_surv
from architecture.densenet import *
from batchgenerators.utilities.file_and_folder_operations import *
from utils.utils import log_class, random_seed_, str2bool
from utils.utils import recursive_find_python_class, tr_val_test, make_surv_array, brk
from tqdm import tqdm
from collections import OrderedDict
from inference import test_data
import random
from architecture.surv_Loss import surv_Loss
from lifelines.utils import concordance_index
from architecture.densenet import *
from architecture.second import Second
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score, concordance_index_ipcw


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=tuple, default=(64, 180, 240))
    parser.add_argument("--out_size", type=int, default=5)
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--output_folder", type=str, default='')
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--backbone", type=str, default='densenet121')
    parser.add_argument("--norm", type=str, default='bn')
    parser.add_argument("--cat", type=str, default='ct')
    return parser.parse_args([])

def load_model(opt, device):
    model_fn = recursive_find_python_class(['architecture'], opt.backbone, current_module='architecture')
    model = model_fn(num_classes=opt.out_size, norm=opt.norm).to(device)
    fname = f"{opt.output_folder}/model_{str(opt.version).zfill(3)}/model_best.model"
    checkpoint = torch.load(fname, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict((key[7:] if key.startswith('module.') else key, value) for key, value in checkpoint['state_dict'].items())
    model.load_state_dict(new_state_dict, strict=False)
    return model.to(device)

def prepare_data(opt):
    df_path = excel_path
    df = pd.read_excel(df_path)
    base_path = ''
    fold = load_json(Fold_path)

    tr_idx = [x for x in fold['0'] + fold['1'] + fold['2'] + fold['3'] + fold['4'] if x not in fold[str(opt.fold)]]
    val_idx = fold[str(opt.fold)]
    test_idx = fold['test']

    tr_data_list = [f"{base_path}/{str(df[df['id'] == int(idx)]['folder_index'].item()).zfill(3)}_{idx}_0000.npy" for idx in tr_idx]
    val_data_list = [f"{base_path}/{str(df[df['id'] == int(idx)]['folder_index'].item()).zfill(3)}_{idx}_0000.npy" for idx in val_idx]
    test_data_list = [f"{base_path}/{str(df[df['id'] == int(idx)]['folder_index'].item()).zfill(3)}_{idx}_0000.npy" for idx in test_idx]

    return tr_data_list, val_data_list, test_data_list, df

def create_dataloaders(opt, tr_data_list, val_data_list, test_data_list, df, breaks):
    train_dataset = Liver_CustomDataset_surv(tr_data_list, df, opt.patch_size, tr_transforms, breaks, img=True)
    val_dataset = Liver_CustomDataset_surv(val_data_list, df, opt.patch_size, val_transforms, breaks, img=True)
    test_dataset = Liver_CustomDataset_surv(test_data_list, df, opt.patch_size, val_transforms, breaks, img=True)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=opt.n_cpu)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=opt.n_cpu)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)
    return train_loader, val_loader, test_loader

def evaluate_model(model, test_loader, device, opt, breaks):
    tb = np.array([np.mean(breaks[i:i+2]) for i in range(len(breaks)-1)])
    test_pred_li = []
    test_death_step = []
    test_death_mo_step = []
    ts_step = []
    td_step = []
    tx_step = []

    with torch.no_grad():
        for i, test_batch in tqdm(enumerate(test_loader)):
            model.eval()
            test_D_ = test_batch['data'].to(device).float()
            test_surv_f = test_batch['surv_f'].long()
            test_surv_s = test_batch['surv_s'].long()
            test_death = test_batch['death'].numpy()
            test_death_mo = test_batch['death_mo'].numpy()
            test_tx = test_batch['tx'].to(device).float()
            test_ft = test_batch['feature'].to(device).float()

            if opt.cat == 'ct':
                test_pred = model(test_D_)
            elif opt.cat == 'tx':
                test_pred = model(test_D_, test_tx)
            elif opt.cat == 'ft':
                test_pred = model(test_D_, test_ft)
            elif opt.cat == 'tx_ft':
                model = Second(num_classes=opt.out_size, norm=opt.norm, nb_cat=6, chn=128, ft=True).to(device)
                test_pred = model(test_D_, (test_tx, test_ft))

            test_pred_li.extend(test_pred.detach().cpu().numpy())
            test_death_step.extend(test_death)
            test_death_mo_step.extend(test_death_mo)
            ts_step.extend(test_surv_s.cpu().numpy())
            td_step.extend(test_surv_f.cpu().numpy())
            tx_step.extend(test_tx.cpu().numpy())

    return test_pred_li, test_death_step, test_death_mo_step, ts_step, td_step, tx_step

def calculate_metrics(all_tr_li, all_li, test_pred_li, tb, time_list):
    rsf_auc_li = []
    br_sc_li = []
    c_li = []

    for ti in time_list:
        pred_li = [np.interp(ti, [0] + list(tb), [1] + list(np.cumprod(t))) for t in test_pred_li]

        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(all_tr_li, all_li, 1 - np.array(pred_li), [ti])
        rsf_auc_li.append(rsf_mean_auc)

        score = brier_score(all_tr_li, all_li, np.array(pred_li), [ti])
        br_sc_li.append(score[1].item())

        c_index = concordance_index_ipcw(all_tr_li, all_li, 1 - np.array(pred_li))[0]
        c_li.append(c_index)

    return rsf_auc_li, br_sc_li, c_li

def main():
    opt = parse_arguments()
    device = torch.device(f'cuda:{opt.gpus}')
    model = load_model(opt, device)
    tr_data_list, val_data_list, test_data_list, df = prepare_data(opt)

    breaks = np.concatenate([np.linspace(0, 90, 16)[:-1], np.array([91])])
    opt.out_size = 15

    train_loader, val_loader, test_loader = create_dataloaders(opt, tr_data_list, val_data_list, test_data_list, df, breaks)

    test_pred_li, test_death_step, test_death_mo_step, ts_step, td_step, tx_step = evaluate_model(model, test_loader, device, opt, breaks)


    all_tr_li = np.array([(d == 1, dm) for d, dm in zip(test_death_step, test_death_mo_step)], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    all_li = np.array([(d == 1, dm) for d, dm in zip(test_death_step, test_death_mo_step)], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    time_list = np.linspace(0, 84, 85)[1:-1]
    rsf_auc_li, br_sc_li, c_li = calculate_metrics(all_tr_li, all_li, test_pred_li, breaks, time_list)

    result = {
        'rsf_auc_li': rsf_auc_li,
        'br_sc_li': br_sc_li,
        'c_index': c_li,
        'test_pred_li': np.array(test_pred_li).tolist(),
        'test_death_step': list(np.array(test_death_step).astype(float)),
        'test_death_mo_step': test_death_mo_step,
        'ts_step': np.array(ts_step).tolist(),
        'td_step': np.array(td_step).tolist(),
        'out_size': opt.out_size,
        'breaks': list(breaks),
        'tx': list(np.where(np.array(tx_step) == 1)[-1].astype(float)),
    }
    save_json(result, f"{opt.output_folder}/result_val.json")

if __name__ == "__main__":
    main()