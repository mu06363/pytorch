import argparse

from train import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=['train', 'test'], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=['on', 'off'], type=str, dest="train_continue")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=10, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./pix2pix_project/datasets/facades", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./pix2pix_project/checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./pix2pix_project/log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./pix2pix_project/result", type=str, dest="result_dir")

parser.add_argument("--task", default="pix2pix", choices=['DCGAN', 'pix2pix'], type=str, dest="task")
parser.add_argument("--opts", nargs="+", default=["direction", 0], dest="opts")

parser.add_argument("--ny", default=256, type=int, dest="ny")
parser.add_argument("--nx", default=256, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--wgt", default=1e2, type=float, dest="wgt")
parser.add_argument('--norm', default='bnorm', choices=['bnorm', 'inorm'], dest='norm')

parser.add_argument("--network", default="pix2pix", choices=['DCGAN', 'pix2pix'], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

if __name__ == '__main__':
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

