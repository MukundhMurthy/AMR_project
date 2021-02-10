# todo define semantics and grammaticality pipeline
import torch
import ipdb
from .models import Transformer
from .main import train_parser
from .preprocess import UniProt_Data
from .utils import generate_mutations, aa_sequence
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import time
import argparse


def cscs_parser(train_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Collect arguments for CSCS evaluation',
                                     parents=[train_parser])
    parser.add_argument('--cached_model_fname', help='filename to load weights')
    parser.add_argument('--POI', help='positions of interest to mutate')
    parser.add_argument('--wt_seqs', help='wildtype sequences for baseline embeddings and probaility prior')
    parser.add_argument('--vocab_file', help="vocab file for interconversion between indices and aa\'s")


class CSCS_objective():
    def __init__(self, model, args, dataset):
        torch.autograd.set_grad_enabled = False
        if model is None:
            model = load_empty_model(args)
        else:
            model.eval()
        self.model = model
        self.positions = args.POI
        self.wt_seqs = args.wt_seqs
        self.mut_seq_dict = generate_mutations
        self.seq_dict = dataset.seq_dict

    def log_embeddings(self):
        # for tokens in self.mut_seq_dict:
        # self.seq_dict[seq]["embedding"] =
        pass

    def compute_semantics(self):
        pass

    def compute_grammar(model, seq_meta):
        pass

    @staticmethod
    def get_aa_sequence(seq):
        return aa_sequence(seq)


def load_empty_model(arg):
    empty_model = Transformer(dataset.vocab_size, hidden=arg.hidden, embed_dim=arg.embed_dim, heads=arg.heads,
                              depth=arg.depth,
                              seq_length=dataset.max_len, drop_prob=arg.drop_prob, mask=True)
    checkpoint = torch.load(arg.state_dict_fname, map_location=torch.device('cpu') if not torch.cuda.is_available()
    else torch.device("cuda: 0"))
    if not torch.cuda.is_available():
        checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")  # remove module.
            checkpoint[name] = v
    model = empty_model.load_state_dict(checkpoint)
    model.eval()
    return empty_model


if __name__ == "__main__":
    train_args = train_parser()
    cscs_args = cscs_parser(train_args)

    dataset = UniProt_Data(filename="uniprot_gpb_rpob", max_len=1600, truncate=True, test=True,
                           job_dir=train_args.job_dir)
