import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep
from pytz import timezone
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import join

import matplotlib
matplotlib.use("agg")  # Use the 'agg' backend for matplotlib

def make_surv_array(t, f, breaks):
    """Generate survival array."""
    n_intervals = len(breaks) - 1
    timegap = np.diff(breaks)
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_intervals * 2))
    if f:
        y_train[0:n_intervals] = 1.0 * (t >= breaks[1:])
        if t < breaks[-1]:
            y_train[n_intervals + np.searchsorted(breaks[1:], t)] = 1
    else:
        y_train[0:n_intervals] = 1.0 * (t >= breaks_midpoint)
    return y_train

def brk(death_all_list, df, br):
    """Generate breaks for survival analysis."""
    death_months = sorted(df[df['id'].isin(death_all_list)]['death_mo'].tolist())
    n = len(death_months) // br
    divided = [death_months[i:i + n] for i in range(0, len(death_months), n)]
    final_breaks = [0] + [group[-1] for group in divided[:-1]] + [91]
    return final_breaks

def acc_cal(pred, target, cls):
    """Calculate accuracy for a specific class."""
    pred = np.array(pred)
    target = np.array(target)
    correct = (pred == target) & (target == cls)
    return correct.sum(), target[target == cls].size

def tr_val_test(data_list):
    """Split data into train, validation, and test sets."""
    np.random.shuffle(data_list)
    n = len(data_list)
    tr_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return data_list[:tr_end], data_list[tr_end:val_end], data_list[val_end:]

class log_class:
    """Log class for training progress and results."""
    def __init__(self, output_folder, csv_file, eval_list):
        self.output_folder = output_folder
        self.csv_file = csv_file
        self.eval_list = eval_list
        self.log_file = None
        self.init_log()

    def init_log(self):
        """Initialize logging and create necessary files."""
        if not os.path.exists(self.csv_file):
            pd.DataFrame({eval_: [None] * 100 for eval_ in self.eval_list}).to_csv(self.csv_file)

    def start_log(self):
        """Start a new log file with the current timestamp."""
        timestamp = datetime.now(timezone('Asia/Seoul'))
        fname = f"training_log_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_file = join(self.output_folder, fname)
        with open(self.log_file, 'w') as file:
            file.write("Starting log...\n")

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        """Print messages to the log file and optionally to the console."""
        timestamp = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
        msg = f"{timestamp}: " + " ".join(str(a) for a in args) + "\n"
        with open(self.log_file, 'a') as file:
            file.write(msg)
        if also_print_to_console:
            print(msg.strip())

def random_seed_(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
