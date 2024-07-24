import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk

from multiprocessing import Pool
from skimage.transform import resize
from batchgenerators.utilities.file_and_folder_operations import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold_path", type=str, help="fold(.json) path")
    parser.add_argument("-e", "--excel_path", type=str, help="excel(.csv) path")
    return parser.parse_args()

def prep(val_id, save_base_path, df, art_base_path, mask_base_path):
    sample_df = df[df['id'] == int(val_id)]
    val_fid = str(sample_df['folder_index'].item()).zfill(3)
    file_path = val_fid + '_' + val_id + '.nii.gz'
    
    art_path = f"{art_base_path}/{file_path}"
    mask_path = f"{mask_base_path}/{file_path}"

    art_itk = sitk.ReadImage(art_path)
    art_npy = sitk.GetArrayFromImage(art_itk)
    seg_itk = sitk.ReadImage(mask_path)
    seg_npy = sitk.GetArrayFromImage(seg_itk)
    seg_npy[seg_npy >= 1] = 1
    art_npy = art_npy * seg_npy
    
    z, x, y = np.where(seg_npy == 1)
    art_z, art_x, art_y = art_npy.shape
    art_npy = art_npy[max(0, min(z)-1):min(max(z)+1, art_z), max(0, min(x)-1):min(max(x)+1, art_x), max(0, min(y)-1):min(max(y)+1, art_y)]
    art_npy = np.clip(art_npy, 0, 185) / 185
    
    new_x = int(art_itk.GetSpacing()[1] * art_x / 1.3476)
    new_y = int(art_itk.GetSpacing()[1] * art_y / 1.3476)
    art_ = resize(art_npy, (art_npy.shape[0], new_x, new_y), order=3, cval=0, mode='edge', anti_aliasing=False)
    
    result = art_[None]
    save_path = f"{save_base_path}/{file_path.split('/')[-1].split('.')[0]}_0000.npy"
    np.save(save_path, result)

def main():
    opt = parse_arguments()

    art_base_path = 'Volume'
    mask_base_path = 'Mask'
    save_data_base_path = 'Preprocessed'
    # Load data
    fold = load_json(opt.fold_path)
    df = pd.read_excel(opt.excel_path)

    all_valid_list = fold['0'] + fold['1'] + fold['2'] + fold['3'] + fold['4'] + fold['test']
    maybe_mkdir_p(save_data_base_path)

    args = zip(all_valid_list, [save_data_base_path]*len(all_valid_list), [df]*len(all_valid_list), [art_base_path]*len(all_valid_list), [mask_base_path]*len(all_valid_list))
    p = Pool(32)
    p.starmap_async(prep, args)
    p.close()
    p.join()

if __name__ == "__main__":
    main()
