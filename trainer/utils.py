import torch
import numpy as np
from torch.utils.data import random_split
from typing import List


def mask_fn(x, mask_diagonal=False):
    # ipdb.set_trace()
    b, h, w = x.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    mask = torch.zeros_like(x)
    mask[:, indices[0], indices[1]] = 1
    final_mask = (mask == 1) & (x == 0)
    x.masked_fill_(final_mask, float('-inf'))
    return x


def train_val_test_split(dataset, mode='random', random_split_frac=(0.2, 0.1), manual_seed=42):
    if mode == 'random':
        test_len = int(np.floor(random_split_frac[0] * len(dataset)))
        valid_len = int(np.floor(random_split_frac[1] * len(dataset)))
        train_len = len(dataset) - (test_len + valid_len)
        return random_split(dataset, [train_len, valid_len, test_len],
                            generator=torch.Generator().manual_seed(manual_seed))


def mutate(seq, pos, id):
    seq[pos] = id
    return seq


def generate_mutations(vocab_len: int, wt_seq: List, positions: List) -> List:
    mutated_seqs_dict = []
    for seq in wt_seq:
        mutated_seqs = [mutate(seq, pos, id) for pos in positions for id in range(1, vocab_len + 1)]
        mutated_seqs_dict[seq] = mutated_seqs
    return mutated_seqs_dict


def aa_sequence(vocab, seq):
    reverse_map = {v: k for k, v in vocab.items()}
    aa_seq_split = [reverse_map[ind] for ind in seq]
    aa_seq = "".join(aa_seq_split)
    return aa_seq