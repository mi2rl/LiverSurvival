from datetime import datetime
from time import time, sleep
from pytz import timezone
import sys
from batchgenerators.utilities.file_and_folder_operations import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pkgutil
import importlib

def tr_val_test(data_list):
    data_list = np.array(data_list)
    tr_data_list = data_list[:int(len(data_list)*0.6)]
    val_data_list = data_list[int(len(data_list)*0.6):int(len(data_list)*0.8)]
    test_data_list = data_list[int(len(data_list)*0.8):]
    return tr_data_list, val_data_list, test_data_list
    
def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr

def random_seed_(seed):
    # random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)