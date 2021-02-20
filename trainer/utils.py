import torch
import numpy as np
from torch.utils.data import random_split
from typing import Dict
from Bio import SeqIO
import json
import ipdb
import datetime
from google.cloud import storage
from collections import OrderedDict
import math


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
        return random_split(dataset, [train_len, valid_len, test_len], generator=torch.Generator().manual_seed(manual_seed))


def mutate(seq, pos, id):
    seq[pos-1] = id
    return "".join(seq)


def generate_mutations(wt_seq_fname, positions_fname) -> Dict:
    # positions is a list of dictionaries, each dictionary is for each wild type sequence, json file"
    aa_mutable_vocab = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V'
    ]
    wt_seqs = read_fasta(wt_seq_fname)
    mutated_seqs_dict = {}
    json_file_path = positions_fname

    with open(json_file_path, 'r') as j:
        positions = json.loads(j.read())

    for i, wt_seq_ in enumerate(wt_seqs):
        wt_seq = str(wt_seq_.seq)
        mutated_seqs_dict[wt_seq] = {}
        for pos_range_label, position_range in positions[wt_seq_.id].items():
            if len(position_range) == 1:
                continue
            else:
                position_range = list(np.arange(position_range[0], position_range[1]))
            for position in position_range:
                for aa in aa_mutable_vocab:
                    mutated_seq = mutate(list(wt_seq), position, aa)
                    mutated_seqs_dict[wt_seq][mutated_seq] = {'cluster': pos_range_label,
                                                              'mut_abbrev': "{0}{1}{2}".format(wt_seq[position-1],
                                                                                               position,
                                                                                               mutated_seq[position-1])}
    return mutated_seqs_dict


def aa_sequence(vocab, seq):
    reverse_map = {v: k for k, v in vocab.items()}
    aa_seq_split = [reverse_map[ind] for ind in seq]
    aa_seq = "".join(aa_seq_split)
    return aa_seq


def read_fasta(fname):
    return list(SeqIO.parse(fname, "fasta"))


def pad(seq, max_len, truncate):
    if len(seq) > max_len and truncate == True:
        seq = seq[:max_len]
    elif len(seq) < max_len:
        seq.extend([0] * (max_len - len(seq)))
    return seq


def tokenize_and_pad(model_type, seqs, vocab, max_len, truncate): #seq_dict used to be a parameter
    padded_aas = []
    for seq in seqs:
        list_aa_indices = [len(vocab)+1] + [vocab[char] for char in seq] #len(vocab)+1 is the starting token
        if model_type == 'bilstm':
            list_aa_indices.insert(0, len(vocab)+2)
            X_pre = [pad(list_aa_indices[:i], max_len, truncate) for i in range(len(list_aa_indices))]
            X_post = [pad(list_aa_indices[i:], max_len, truncate) for i in range(len(list_aa_indices))]
            padded_aas.append([X_pre, X_post])
        else:
            padded_aa_indices = pad(list_aa_indices, max_len, truncate)
        # seq_dict[seq]["tokens"] = padded_aa_indices
            padded_aas.append(padded_aa_indices)
    return padded_aas


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
    fname = "{0}_{1}.pth".format(arg.name_run, arg.epochs)
    return fname


def download_from_gcloud_bucket(fname):
    storage_client = storage.Client()
    public_bucket = storage_client.bucket('amr-transformer')
    blob = public_bucket.blob(fname)
    blob.download_to_filename(fname)
    fname = "./{0}".format(fname)
    return fname


def mut_abbrev_to_seq(abbrev, wt):
    wt = list(wt)
    pos = int(abbrev[1:-1])
    aa = abbrev[-1]
    wt[pos-1] = aa
    wt = "".join(wt)
    return wt


def generate_vocab(forbidden_aa):
    amino_acids = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
    for aa in forbidden_aa:
        amino_acids.remove(aa)
    vocab = OrderedDict({aa: idx + 1 for idx, aa in enumerate(amino_acids)})
    return vocab


def calc_bedroc(scores, alpha):
    #scores must be ranked
    #Implementation copied from rdkit. We avoided installing rdkit as it would have to be the only conda installation.
    num_seq = len(scores)
    denom = 1.0 / num_seq * ((1 - math.exp(-alpha)) / (math.exp(alpha / num_seq) - 1))

    num_escape = 0
    sum_exp = 0

    for i in range(num_seq):
        escape = scores[i] == 1
        if escape:
            num_escape += 1
            sum_exp += math.exp(-(alpha * (i + 1)) / num_seq)

    assert num_escape > 0
    rie = sum_exp/(num_escape * denom)

    ratio = 1.0 * num_escape / num_seq
    rie_max = (1 - math.exp(-alpha * ratio)) / (ratio * (1 - math.exp(-alpha)))
    rie_min = (1 - math.exp(alpha * ratio)) / (ratio * (1 - math.exp(alpha)))

    if rie_max != rie_min:
        bedroc = (rie - rie_min) / (rie_max - rie_min)
    else:
        bedroc = 1.0

    return bedroc






if __name__ == '__main__':
    hi = generate_mutations('../escape_validation/anchor_seqs.fasta', '../escape_validation/regions_of_interest.json')
    ipdb.set_trace(
    )

    # avail_mut_types = ['primary', 'compensatory']
    # avail_comp_eval_type = ['single, combinatoric']
    # avail_genes = ['rpoB']
    # avail_types = [avail_mut_types, avail_comp_eval_type, avail_genes]
    # params = [mutation_types, compensatory_evaluation_types, gene]
    # for param, avail_type in zip(params, avail_types):
    #     if len(param) == 1:
    #         if param not in avail_type:
    #             print("{0} not in {1}".format(param, avail_type))
    #     else:
    #         for single_param in param:
    #             if single_param not in avail_type:
    #                 print("{0} not in {1}".format(single_param, avail_type))