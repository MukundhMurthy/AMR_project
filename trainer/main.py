# coding: utf-8
import torch
import torch.nn as nn
import ipdb
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import wandb
import uuid
import os
import numpy as np
import logging
import datetime
from google.cloud import storage
from .models import Transformer
from trainer.utils import train_val_test_split
from trainer.preprocess import UniProt_Data
import sys
# from Bio.Seq import Seq
# from torch.nn.functional import one_hot
# from torch.autograd import Variable
# import numpy as np


def train(arg, data):
    print(arg.wandb)

    cuda_availability = torch.cuda.is_available()
    if cuda_availability:
        # os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        logging.info("The default device is {0}".format(device))
    else:
        device = 'cpu'

    torch.backends.cudnn.benchmark = True

    train_dataset, valid_dataset, test_dataset = train_val_test_split(data, mode=arg.split_method,
                                                                      manual_seed=arg.manual_seed)
    loader_params = {
        'batch_size': arg.batch_size,
        'shuffle': True,
        'num_workers': arg.num_workers,
        'drop_last': True
    }
    test_loader = DataLoader(test_dataset, **loader_params)
    train_loader = DataLoader(train_dataset, **loader_params)
    valid_loader = DataLoader(valid_dataset, **loader_params)
    ori_model = Transformer(dataset.vocab_size, hidden=arg.hidden, embed_dim=arg.embed_dim, heads=arg.heads,
                        depth=arg.depth,
                        seq_length=dataset.max_len, drop_prob=arg.drop_prob, mask=True)

    if torch.cuda.is_available():
        model = ori_model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info("The total number of GPUs is {0}".format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0, 1, 2, 3])

    opt = torch.optim.Adam(lr=arg.learning_rate, params=model.parameters())
    if arg.lr_scheduler is not None:
        if arg.lr_scheduler == 'lambda_lr':
            raise NotImplementedError
        if arg.lr_scheduler == 'plateau':
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=arg.patience, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=0)
    count = 0
    no_improvement = 0
    avg_valid_loss_list = []

    for e in range(arg.epochs):
        training_loss_list = []
        valid_loss_list = []
        test_loss_list = []
        for t, (X, y) in enumerate(train_loader):
            model.train()
            X, y = X.to(device), y.to(device)
            out = model(X.long()).transpose(1, 2)
            training_loss = criterion(out, y.long())
            opt.zero_grad()
            training_loss.backward()
            opt.step()
            # sch.step(training_loss)
            training_loss_list.append(training_loss.item())
            if t % arg.within_epoch_interval == 0:
                logging.info('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, training_loss.item()))
                if arg.wandb:
                    wandb.log({"intra-epoch training loss": training_loss.item()})
                count += 1
        with torch.no_grad():
            model.eval()
            for batch_idx, (valid_x, valid_y) in enumerate(valid_loader):
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                out = model(valid_x.long()).transpose(1, 2)
                valid_loss = criterion(out, valid_y.long())
                valid_loss_list.append(valid_loss.item())
            for batch_idx, (test_x, test_y) in enumerate(test_loader):
                test_x, test_y = test_x.to(device), test_y.to(device)
                out = model(test_x.long()).transpose(1, 2)
                test_loss = criterion(out, test_y.long())
                test_loss_list.append(test_loss.item())

        final_valid_loss = np.array(valid_loss_list).mean()
        avg_valid_loss_list.append(final_valid_loss)
        # Early stopping Implementation
        print('{0}, {1}'.format(final_valid_loss, min(valid_loss_list)))
        if final_valid_loss > min(avg_valid_loss_list):
            no_improvement += 1
            if no_improvement == arg.es_patience:
                logging.info("Early stopping at epoch {0}".format(e))
                torch.save(model.state_dict(), "{0}_{1}.pth".format(arg.name_run, e))
                if args.wandb:
                    wandb.save("{0}_{1}.pth".format(arg.name_run, e))
                return ori_model
                sys.exit(0)
        else:
            no_improvement = 0
        final_train_loss = np.array(training_loss_list).mean()
        final_test_loss = np.array(test_loss_list).mean()
        if arg.lr_scheduler == 'plateau':
            sch.step(final_valid_loss)

        if arg.wandb:
            wandb.log({'training_loss': final_train_loss, 'valid_loss': final_valid_loss, 'test_loss': final_test_loss,
                       'epoch': e})
        logging.info("For epoch {0}, the mean training loss = {1}, mean validation loss = {2}, mean test_loss = {3}"
                     .format(e, final_train_loss, final_valid_loss, final_test_loss))

        if e % 5 == 0:
            torch.save(model.state_dict(), "{0}_{1}.pth".format(arg.name_run, e))
            if arg.wandb:
                wandb.save("{0}_{1}.pth".format(arg.name_run, e))

    torch.save(model.state_dict(), "{0}_{1}.pth".format(arg.name_run, arg.epochs))

    if arg.wandb:
        wandb.run.summary['test_loss'] = max(test_loss_list)
        wandb.save("{0}_{1}.pth".format(arg.name_run, arg.epochs))

    if arg.job_dir:
        save_model(args)

    return ori_model


def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training set")
    parser.add_argument("--hidden", type=int, default=16, help="# of hidden layers for transformer block")
    parser.add_argument("--embed_dim", type=int, default=16, help="embedding dimension for transformer")
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
    return parser.parse_args()


def save_model(arg):
    """Saves the model to Google Cloud Storage
    Args:
      arg: contains name for saved model.
    """
    scheme = 'gs://'
    bucket_name = arg.job_dir[len(scheme):].split('/')[0]

    prefix = '{}{}/'.format(scheme, bucket_name)
    bucket_path = arg.job_dir[len(prefix):].rstrip('/')

    datetime_ = datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S')

    if bucket_path:
        model_path = '{}/{}/{}'.format(bucket_path, datetime_, arg.name_run)
    else:
        model_path = '{}/{}'.format(datetime_, arg.name_run)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename("{0}_{1}.pth".format(arg.name_run, arg.epochs))


if __name__ == '__main__':
    # datetime_ = datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S')
    # , filename = '{0}.log'.format(datetime_)
    logging.basicConfig(level=logging.INFO)
    # todo build out argument parser
    args = train_parser()
    if args.wandb:
        wandb_configured = wandb.login(key="f21ac4792d9b90f6ddae4a3964556d2686fbfe91")
        print("Log in to wandb successful: {0}".format(wandb_configured))
        os.environ['WANDB_MODE'] = 'run'
        unique_run_id = uuid.uuid1().hex[0:8]
        wandb.init(config={"Unique Run ID": unique_run_id}, project='Antimicrobial resistance', name=args.name_run)
        wandb.config.update(args)
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    dataset = UniProt_Data(filename="uniprot_gpb_rpob", max_len=1600, truncate=True, test=True, job_dir=args.job_dir)
    if args.wandb:
        wandb.save("vocab.json")
    model = train(args, dataset)

