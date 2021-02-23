import torch
import logging
from .lm_models import Transformer
from .utils import train_val_test_split, save_model
import numpy as np
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader


def train(arg, dataset):
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

    train_dataset, valid_dataset, test_dataset = train_val_test_split(dataset, mode=arg.split_method,
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
    model = Transformer(dataset.vocab_size, hidden=arg.hidden, embed_dim=arg.embed_dim, heads=arg.heads,
                        depth=arg.depth,
                        seq_length=dataset.max_len, drop_prob=arg.drop_prob, mask=True)

    model.to(device)

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
                if arg.wandb:
                    wandb.save("{0}_{1}.pth".format(arg.name_run, e))
                fname = "{0}_{1}.pth".format(arg.name_run, e)
                return fname, model
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
    fname = "{0}_{1}.pth".format(arg.name_run, arg.epochs)
    if arg.wandb:
        wandb.run.summary['test_loss'] = max(test_loss_list)
        wandb.save("{0}_{1}.pth".format(arg.name_run, arg.epochs))

    if arg.job_dir:
        fname = save_model(arg)

    return fname, model

