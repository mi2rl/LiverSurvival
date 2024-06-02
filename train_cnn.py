import argparse
import torch
import torch.nn as nn
from augment import tr_transforms, val_transforms
import pandas as pd
import numpy as np
from time import time
from torch.utils.data import DataLoader
from dataloader import Liver_CustomDataset_surv
from architecture.densenet import *
from batchgenerators.utilities.file_and_folder_operations import subfiles, maybe_mkdir_p, load_json, save_json
from utils.utils import log_class, random_seed_, recursive_find_python_class
from tqdm import tqdm
from collections import OrderedDict
from architecture.surv_Loss import surv_Loss
from lifelines.utils import concordance_index

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epoch", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--patch_size", type=tuple, default=(64, 180, 240))
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--out_size", type=int, default=5)
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--output_folder", type=str, default='')
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--backbone", type=str, default='densenet121')
    parser.add_argument("--norm", type=str, default='bn')
    parser.add_argument("--br", type=int, default=5)
    parser.add_argument("--cat", type=str, default='ct')
    parser.add_argument("--gpu_1", type=int, default=0)
    parser.add_argument("--gpu_2", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    return parser.parse_args()

def prepare_data(opt):
    df_path = '/workspace/src/Liver/CDSS_Liver/tx_data_excel.xlsx'
    df = pd.read_excel(df_path)

    valid_id_list = [f.split('_')[-1].split('.')[0] for f in subfiles('Volume_5mm', join=False, suffix='.img')]
    death_all_list = []

    for valid_id in tqdm(valid_id_list):
        sample_df = df[df['id'] == int(valid_id)]
        if sample_df['Tx_LT'].item() == 1:
            continue
        if sample_df['Asan/Out'].item() == 'Out':
            continue
        if sample_df['death_01'].item() == 1:
            death_all_list.append(valid_id)

    breaks = get_breaks(opt.br, death_all_list, df)
    opt.out_size = len(breaks) - 1

    tr_data_list, val_data_list, test_data_list = get_data_lists(opt, df)
    
    return df, breaks, tr_data_list, val_data_list, test_data_list

def get_breaks(br, death_all_list, df):
    def brk(death_all_list, df, br):
        g_li = [df[df['id'] == int(d)]['death_mo'].item() for d in death_all_list]
        g_li.sort()
        n = len(g_li) // br
        result = [g_li[i:i + n] for i in range(0, len(g_li), n)]
        final = [0] + [r[-1] for r in result[:-2]] + [91]
        return final

    if br == 5:
        return np.array([0, 3.3, 8.3, 20, 41, 91])
    elif br == 1:
        return np.concatenate([np.linspace(0, 91, 92)[:-1], np.array([91])])
    elif br == 3:
        return np.concatenate([np.linspace(0, 90, 31)[:-1], np.array([91])])
    elif br == 6:
        return np.concatenate([np.linspace(0, 90, 16)[:-1], np.array([91])])
    elif br == 12:
        return np.concatenate([np.linspace(0, 84, 8)[:-1], np.array([91])])
    elif br == 18:
        return np.concatenate([np.linspace(0, 72, 5)[:], np.array([91])])
    elif br == 24:
        return np.concatenate([np.linspace(0, 72, 4)[:], np.array([91])])
    elif br == 36:
        return np.array([0., 36., 72., 91.])
    elif br == 7:
        return np.round(brk(death_all_list, df, br), 2)
    elif br == 15:
        return np.round(brk(death_all_list, df, br), 2)
    elif br == 30:
        return np.round(brk(death_all_list, df, br), 2)
    elif br == 91:
        breaks = np.round(brk(death_all_list, df, 91), 2)
        return np.concatenate([breaks[:-14], breaks[-14::2]])

def get_data_lists(opt, df):
    base_path = ''
    fold = load_json(Fold_path)

    tr_idx = [x for x in fold['0'] + fold['1'] + fold['2'] + fold['3'] + fold['4'] if x not in fold[str(opt.fold)]]
    val_idx = fold[str(opt.fold)]
    test_idx = fold['test']

    def get_paths(idx_list):
        return [f"{base_path}/{str(df[df['id'] == int(idx)]['folder_index'].item()).zfill(3)}_{idx}_0000.npy" for idx in idx_list]

    return get_paths(tr_idx), get_paths(val_idx), get_paths(test_idx)

def initialize_model(opt, device):
    if opt.cat == 'ct':
        model_fn = recursive_find_python_class(['architecture'], opt.backbone, current_module='architecture')
        model = model_fn(num_classes=opt.out_size, norm=opt.norm).to(device)
    elif opt.cat == 'tx':
        model_fn = recursive_find_python_class(['architecture_concat'], opt.backbone, current_module='architecture_concat')
        model = model_fn(num_classes=opt.out_size, norm=opt.norm, nb_cat=6, chn=128, ft=False).to(device)
    elif opt.cat == 'ft':
        model_fn = recursive_find_python_class(['architecture_concat'], opt.backbone, current_module='architecture_concat')
        model = model_fn(num_classes=opt.out_size, norm=opt.norm, nb_cat=14, chn=128, ft=False).to(device)
    elif opt.cat == 'tx_ft':
        model_fn = recursive_find_python_class(['architecture_concat'], opt.backbone, current_module='architecture_concat')
        model = model_fn(num_classes=opt.out_size, norm=opt.norm, nb_cat=6, chn=128, ft=True).to(device)

    model = nn.DataParallel(model, device_ids=[opt.gpu_1, opt.gpu_2])
    return model

def make_surv_array(t, f, breaks):
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_intervals * 2))
    if f:
        y_train[:n_intervals] = 1.0 * (t >= breaks[1:])
        if t < breaks[-1]:
            y_train[n_intervals + np.where(t < breaks[1:])[0][0]] = 1
    else:
        y_train[:n_intervals] = 1.0 * (t >= breaks_midpoint)
    return y_train

def prepare_dataloaders(tr_data_list, val_data_list, test_data_list, df, opt, breaks):
    train_dataset = Liver_CustomDataset_surv(tr_data_list, df, opt.patch_size, tr_transforms, breaks, img=True)
    val_dataset = Liver_CustomDataset_surv(val_data_list, df, opt.patch_size, val_transforms, breaks, img=True)
    test_dataset = Liver_CustomDataset_surv(test_data_list, df, opt.patch_size, val_transforms, breaks, img=True)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, criterion, optimizer, train_loader, device, opt):
    model.train()
    loss_step, prod_step, death_step, death_mo_step = [], [], [], []
    for batch in train_loader:
        optimizer.zero_grad()

        D_ = batch['data'].to(device).float()
        surv_f = batch['surv_f'].to(device).long()
        surv_s = batch['surv_s'].to(device).long()
        death = batch['death'].numpy()
        death_mo = batch['death_mo'].numpy()
        tx = batch['tx'].to(device).float()
        ft = batch['feature'].to(device).float()

        if opt.cat == 'ct':
            pred = model(D_)
        elif opt.cat == 'tx':
            pred = model(D_, tx)
        elif opt.cat == 'ft':
            pred = model(D_, ft)
        elif opt.cat == 'tx_ft':
            model = Second(num_classes=opt.out_size, norm=opt.norm, nb_cat=6, chn=128, ft=True).to(device)
            pred = model(test_D_, (test_tx, test_ft))
        Loss_ = criterion(surv_s, surv_f, pred)
        Loss_.backward()
        optimizer.step()

        loss_step.append(Loss_.cpu().item())
        prod_step.extend(np.cumprod(pred.detach().cpu().numpy(), axis=1)[:, -1])
        death_step.extend(death)
        death_mo_step.extend(death_mo)

    return np.mean(loss_step), prod_step, death_step, death_mo_step

def validate_one_epoch(model, criterion, val_loader, device, opt):
    model.eval()
    val_loss_step, val_prod_step, val_death_step, val_death_mo_step = [], [], [], []
    with torch.no_grad():
        for val_batch in val_loader:
            val_D_ = val_batch['data'].to(device).float()
            val_surv_f = val_batch['surv_f'].to(device).long()
            val_surv_s = val_batch['surv_s'].to(device).long()
            val_death = val_batch['death'].numpy()
            val_death_mo = val_batch['death_mo'].numpy()
            val_tx = val_batch['tx'].to(device).float()
            val_ft = val_batch['feature'].to(device).float()

            if opt.cat == 'ct':
                val_pred = model(val_D_)
            elif opt.cat == 'tx':
                val_pred = model(val_D_, val_tx)
            elif opt.cat == 'ft':
                val_pred = model(val_D_, val_ft)
            elif opt.cat == 'tx_ft':
                model = Second(num_classes=opt.out_size, norm=opt.norm, nb_cat=6, chn=128, ft=True).to(device)
                test_pred = model(test_D_, (test_tx, test_ft))

            val_Loss_ = criterion(val_surv_s, val_surv_f, val_pred)

            val_loss_step.append(val_Loss_.cpu().item())
            val_prod_step.extend(np.cumprod(val_pred.detach().cpu().numpy(), axis=1)[:, -1])
            val_death_step.extend(val_death)
            val_death_mo_step.extend(val_death_mo)

    return np.mean(val_loss_step), val_prod_step, val_death_step, val_death_mo_step

def log_results(epoch, loss, val_loss, Cindex, val_Cindex, Log):
    Log.print_to_log_file(f"epoch = {epoch + 1}")
    Log.print_to_log_file(f"  * tr_loss = {loss}")
    Log.print_to_log_file(f"  * val_loss = {val_loss}")
    Log.print_to_log_file(f"  * tr_Cindex = {Cindex}")
    Log.print_to_log_file(f"  * val_Cindex = {val_Cindex}")

def save_best_model(epoch, model, optimizer, loss_epoch, val_loss_epoch, Cindex, val_Cindex, result_folder):
    fname = f"{result_folder}/model_best.model"
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    optimizer_state_dict = optimizer.state_dict()
    save_this = {
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss_epoch,
        'loss_val': val_loss_epoch,
        'Cindex': Cindex,
        'val_Cindex': val_Cindex
    }
    torch.save(save_this, fname)

def main():
    opt = parse_arguments()
    random_seed_(opt.random_seed)
    device = torch.device(f'cuda:{opt.gpus}')

    df, breaks, tr_data_list, val_data_list, test_data_list = prepare_data(opt)
    model = initialize_model(opt, device)
    criterion = surv_Loss(device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    
    train_loader, val_loader, _ = prepare_dataloaders(tr_data_list, val_data_list, test_data_list, df, opt, breaks)

    result_folder = f"{opt.output_folder}/model_{str(opt.version).zfill(3)}"
    maybe_mkdir_p(result_folder)
    Log = log_class(result_folder, f"{opt.output_folder}.csv", ['cindex_5', 'cindex_10', 'cindex_15', 'cindex_20', 'cindex_25', 'cindex_30'])
    Log.start_log()
    Log.print_to_log_file(opt)

    t0 = time()
    best_val_Cindex = 0

    for epoch in range(opt.nb_epoch):
        train_loss, train_prod, train_death, train_death_mo = train_one_epoch(model, criterion, optimizer, train_loader, device, opt)
        val_loss, val_prod, val_death, val_death_mo = validate_one_epoch(model, criterion, val_loader, device, opt)

        train_Cindex = concordance_index(train_death_mo, train_prod, train_death)
        val_Cindex = concordance_index(val_death_mo, val_prod, val_death)

        log_results(epoch, train_loss, val_loss, train_Cindex, val_Cindex, Log)

        if val_Cindex > best_val_Cindex:
            save_best_model(epoch, model, optimizer, train_loss, val_loss, train_Cindex, val_Cindex, result_folder)
            best_val_Cindex = val_Cindex

        Log.print_to_log_file(f'Training time for one epoch : {time() - t0:.1f}')
        Log.plot_progress(epoch, (train_Cindex, val_Cindex), (train_loss, val_loss))

    Log.print_to_log_file(f'Total training time : {time() - t0:.1f}')

if __name__ == "__main__":
    main()
