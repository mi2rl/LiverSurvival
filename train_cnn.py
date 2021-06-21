import os
import argparse

import pytorch_lightning as pl

from common import build_parser
from model.cnn import CNNSurvival


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
    transform_list = []
    # data_module = 

    # ------------
    # model
    # ------------
    model = CNNSurvival()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data_module)



if __name__ == '__main__':
    main()