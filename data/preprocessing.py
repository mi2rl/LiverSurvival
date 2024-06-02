import shutil
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from scipy.ndimage.interpolation import map_coordinates
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from batchgenerators.augmentations.utils import resize_segmentation
import pandas as pd


def prep(val_id, save_base_path, df, art_base_path, mask_base_path):
    sample_df = df[df['id'] == int(val_id)]
    val_fid = str(sample_df['folder_index'].item()).zfill(3)
    file_path = val_fid + '_' + val_id + '.img'
    
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

# User-defined variables
fold_path = 'Fold path'
excel_path = 'Excel path'
art_base_path = 'Volume_5mm'
mask_base_path = '_New_mask'
save_data_base_path = 'Save data base path'

# Load data
fold = load_json(fold_path)
df = pd.read_excel(excel_path)

all_valid_list = fold['0'] + fold['1'] + fold['2'] + fold['3'] + fold['4'] + fold['test']
maybe_mkdir_p(save_data_base_path)

args = zip(all_valid_list, [save_data_base_path]*len(all_valid_list), [df]*len(all_valid_list), [art_base_path]*len(all_valid_list), [mask_base_path]*len(all_valid_list))
p = Pool(32)
p.starmap_async(prep, args)
p.close()
p.join()
