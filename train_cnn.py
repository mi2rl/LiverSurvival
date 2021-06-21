import os
import argparse

import pytorch_lightning as pl

from common import build_parser
from model.cnn import CNNSurvival
from argparse import ArgumentParser
from liver import load_dataloader
from transforms import tr_transforms, val_transforms

def build_args(parser):
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CNNSurvival.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def main():
    pl.seed_everything(1234)
    args = build_args(build_parser())

    # ------------
    # data
    # ------------
    df_path = '/workspace/src/CDSS_Liver/tx_data_excel.xlsx'
    data_path = '/workspace/src/1_Classification/data_preprocessed_liver'
    dataloader, val_dataloader, test_dataloader = \
    load_dataloader(df_path, data_path, tr_transforms, val_transforms, args.batch_size_tr, args.batch_size_val, args.n_cpu)
    # ------------
    # model
    # ------------
    model = CNNSurvival(args.out_size, args.norm)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data_module)



if __name__ == '__main__':
    main()