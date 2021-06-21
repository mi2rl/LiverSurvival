import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epoch", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--batch_size_tr", type=int, default=2, help="size of the batches")
    parser.add_argument("--batch_size_val", type=int, default=2, help="size of the batches")
    parser.add_argument("--patch_size", type=tuple, default=(96, 192, 192))
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--out_size", type=int, default=2)
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--output_folder", type=str, default='./liver_result')
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--backbone", type=str, default='densenet121')
    parser.add_argument("--norm", type=str, default='in')

    return parser