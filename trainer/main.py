# coding: utf-8
from argparse import ArgumentParser
import wandb
import uuid
import os
import logging
from .preprocess import UniProt_Data
from .train import train
from .cscs import CSCS_objective
import torch
import datetime
import ipdb
# from Bio.Seq import Seq
# from torch.nn.functional import one_hot
# from torch.autograd import Variable
# import numpy as np


def train_parser():
    parser = ArgumentParser(description="parent parser for model training", add_help=False)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training set")
    parser.add_argument("--hidden", type=int, default=16, help="# of hidden layers for transformer block")
    parser.add_argument("--embed_dim", type=int, default=16, help="embedding dimension for transformer")
    parser.add_argument("--min_len", type=int, default=1200, help="minimum length (e.g. to avoid fragments)")
    parser.add_argument("--heads", type=int, default=2, help="# of heads for transformer")
    parser.add_argument("--depth", type=int, default=1, help='# of transformer block repeats')
    parser.add_argument("--drop_prob", type=float, default=0.1, help='Dropout for transformer blocks')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--model_type", choices=['attention'], default='attention', help="model to pretrain")
    parser.add_argument("--lr_scheduler", choices=['plateau', 'adaptive'], default='plateau')
    parser.add_argument("--within_epoch_interval", type=int, default=5, help="show update ever # of bactches")
    parser.add_argument("--patience", type=int, default=1, help="num epochs for lr scheduler to wait before decreasing")
    parser.add_argument("--wandb", action="store_true", help="log results using wandb", default=False)
    parser.add_argument('--name_run', type=str, help="name of run to save in wandb")
    parser.add_argument('--split_method', type=str, help="method to split dataset. Random available.", default='random')
    parser.add_argument('--manual_seed', type=int, help="integer seed for random split torch generator", default=42)
    parser.add_argument('--num_workers', type=int, help="number of cpu workers for dataloaders", default=0)
    parser.add_argument('--job-dir', help='GCS location to export models')
    parser.add_argument('--es_patience', type=int, default=2, help='Early stopping patience')
    return parser


def cscs_parser(train_parser):
    parser = ArgumentParser(description='Collect arguments for CSCS evaluation',
                                     parents=[train_parser])
    parser.add_argument('--state_dict_fname', help='filename to load weights')
    parser.add_argument('--POI_file', help='positions of interest file')
    parser.add_argument('--wt_seqs_file', help='wildtype sequences for baseline embeddings and probability prior')
    parser.add_argument('--eval_batch_size', help='batch_size for model evaluation')
    parser.add_argument('--cscs_debug', type=bool, default=False, help='debug_mode')
    # parser.add_argument('--vocab_file', help="vocab file for interconversion between indices and aa\'s")
    return parser


if __name__ == '__main__':
    # datetime_ = datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S')
    # , filename = '{0}.log'.format(datetime_)
    logging.basicConfig(level=logging.INFO)
    # todo build out argument parser

    parser = train_parser()
    cscs_parser = cscs_parser(parser)
    args = cscs_parser.parse_args()

    if args.wandb:
        wandb_configured = wandb.login(key="f21ac4792d9b90f6ddae4a3964556d2686fbfe91")
        print("Log in to wandb successful: {0}".format(wandb_configured))
        os.environ['WANDB_MODE'] = 'run'
        unique_run_id = uuid.uuid1().hex[0:8]
        wandb.init(config={"Unique Run ID": unique_run_id}, project='Antimicrobial resistance', name=args.name_run)
        wandb.config.update(args)
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    dataset = UniProt_Data(filename="uniprot_gpb_rpob", min_len=args.min_len, max_len=None, truncate=True, test=args.cscs_debug, job_dir=args.job_dir)
    if args.wandb:
        wandb.save("vocab.json")

    state_dict_fname, model = train(args, dataset)
    ipdb.set_trace()
    cscs_computer = CSCS_objective(args, dataset, model=model, cscs_debug=args.cscs_debug)
    cscs_computer = cscs_computer.compute_semantics()
    mut_seq_dict = cscs_computer.compute_grammar()

    datetime_ = datetime.datetime.now().strftime('cscs_%Y%m%d_%H%M%S')
    state_dict = torch.load(state_dict_fname)
    state_dict.update(mut_seq_dict)
    torch.save(state_dict, state_dict_fname)
    # json.dump(
    #     mut_seq_dict,
    #     open("{0}.json".format(datetime_), "w"))

    if args.wandb:
        wandb.save(state_dict_fname)




