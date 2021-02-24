from Bio import SeqIO
import json
import logging
import re
from torch.utils.data import Dataset
import torch
from google.cloud import storage
from .utils import tokenize_and_pad, read_fasta, download_from_gcloud_bucket
from collections import OrderedDict
from itertools import chain
import ipdb


class Preprocesser:
    def __init__(self, fname, condition, model_type='attention', min_len=None, max_len=None, truncate=False, forbidden_aas=('X'),
                 debug_mode=False, job_dir=None):
        if forbidden_aas is None:
            forbidden_aas = ['X']
        # root_dir = __file__.split('/')[:-1]
        # root_dir = '/'.join(root_dir)

        # if not arg.job_dir:
        if job_dir is None:
            fname = '{0}.fasta'.format(fname)
        else:
            fname = download_from_gcloud_bucket("{0}.fasta".format(fname))
        # else:

        self.records = read_fasta(fname)
        # except:
        #     fname = root_dir + '/{0}.fasta'.format(fname)
        #     self.records = list(SeqIO.parse(fname, "fasta"))

        print(fname)
        self.debug_mode = debug_mode
        self.val = lambda x: 500 if self.debug_mode else len(x) + 1
        self.truncate = truncate
        self.max_len = max_len
        self.min_len = min_len
        self.fname = fname
        self.model_type = model_type
        self.val = 1 if self.model_type == 'bilistm' else self.val
        self.condition = condition  # tuple (species, value) or (identifier, value)
        self.metas = self.save_meta()
        self.forbidden_aas = list(forbidden_aas)
        self.seq_dict, self.num_seqs = self.collect_sequences()
        self.seqs = list(self.seq_dict.keys())
        if max_len is None:
            self.max_len = max([len(seq) for seq in self.seqs])
        if self.model_type == 'attention':
            additional_tokens = 1
        elif self.model_type == 'bilstm':
            additional_tokens = 2

        self.max_len += additional_tokens
        if self.min_len is not None:
            self.min_len += additional_tokens

        amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
            'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
        ]

        for aa in self.forbidden_aas:
            amino_acids.remove(aa)
        self.vocab = OrderedDict({aa: idx + 1 for idx, aa in enumerate(amino_acids)})
        # jsn = json.dumps(self.vocab)
        # f = open("vocab.json", "w")
        # f.write(jsn)
        # f.close()

    def collect_sequences(self):
        seqs = {}
        for record in self.records[:self.val(self.records)]:
            cond1 = "X" in record.seq
            access = self.get_access(record.description)[0]
            meta_info = self.metas[access]
            if self.condition is not None:
                cond2 = meta_info[self.condition[0]] != self.condition[1]
            else:
                cond2 = False
            if self.max_len is None:
                cond3 = False
            else:
                cond3 = self.truncate==False and len(record.seq) > self.max_len
            if self.min_len is None:
                cond4 = False
            else:
                cond4 = len(record.seq) < self.min_len
            if cond1 or cond2 or cond3 or cond4:
                continue
            meta_info['seq_len'] = len(record.seq)
            seqs[record.seq] = {}
            seqs[record.seq]["meta_info"] = meta_info
            # seqs[record.seq].append(meta_info)
        print('goo')
        if self.condition is not None:
            logging.info("After elimination of sequences with {0} forbidden amino acids and selection of sequences where {1} \
            is equal to {2}, {3} sequences are passed for training").format("".join([aa + ' ,' for aa in \
                                                                                     self.forbidden_aas]),
                                                                            self.condition[0], self.condition[1],
                                                                            len(seqs.keys()))
        else:
            logging.info("After elimination of sequences with {0} forbidden amino acids. {1} "
                         "sequences are passed for training"
                         .format("".join([aa + ' ,' for aa in self.forbidden_aas]), len(seqs.keys())))
        return seqs, len(seqs.keys())

    @staticmethod
    def get_access(description):
        return re.findall('\|(.+?)\|', description)

    def save_meta(self):
        metas = {}
        with open(self.fname) as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                full_line = line[1:].rstrip()
                accession = self.get_access(full_line)[0]
                metas[accession] = {
                    'protein_entry': re.findall('(?<=....\|).*?(?=\s)', full_line)[0],
                    'gene_entry': re.findall('(?<=\s).*?(?=\sOS)', full_line)[0],
                    'organism_name': re.findall('(?<=OS\=).*?(?=\sOX\=)', full_line)[0],
                    'organism_identifier': re.findall('(?<=OX\=).*?(?=\sGN\=)', full_line)[0],
                    'gene_name': re.findall('(?<=GN\=).*?(?=\sPE\=)', full_line)[0],
                    'protein_existence': re.findall('(?<=PE\=)\d(?=\sSV\=)', full_line)[0],
                    'sequence_version': re.findall('(?<=SV\=).*$', full_line)[0]
                }
        return metas

    def X_y_from_seq(self):
        # ipdb.set_trace()
        tokenized = tokenize_and_pad(self.model_type, self.seqs, self.vocab, self.max_len, self.truncate)
        if self.model_type == 'attention':
            tokenized_tensor = torch.Tensor(tokenized)
        # assert tokenized_tensor.size() == self.num_seqs, self.max_len
            X = tokenized_tensor[:, :-1]
            y = tokenized_tensor[:, 1:]
            return X, y
        elif self.model_type == 'bilstm':
            list_x_pre = list(chain.from_iterable([tokens_list[0] for tokens_list in tokenized]))
            list_x_post = list(chain.from_iterable([tokens_list[1] for tokens_list in tokenized]))
            list_y = list(chain.from_iterable([tokens_list[2] for tokens_list in tokenized]))

            x_pre_tensor = torch.Tensor(list_x_pre)
            x_post_tensor = torch.Tensor(list_x_post)
            y_tensor = torch.Tensor(list_y)

            return x_pre_tensor, x_post_tensor, y_tensor


        # print(X.size(), y.size())
        # y = one_hot(tokenized_tensor[:, 1:].to(torch.int64), num_classes=len(self.vocab))



class UniProt_Data(Dataset):
    def __init__(self, model_type='attention', condition=None, min_len=None, max_len=None, truncate=False, forbidden_aas=('X'),
                 filename="uniprot_gpb_rpob", test=False, job_dir=None):
        super().__init__()
        preprocess = Preprocesser(filename, condition, model_type=model_type, min_len=min_len, max_len=max_len, truncate=truncate,
                                  forbidden_aas=forbidden_aas, debug_mode=test, job_dir=job_dir)
        self.model_type = model_type
        self.seqs = preprocess.seqs
        self.max_len = preprocess.max_len
        if self.model_type == 'attention':
            self.X, self.y = preprocess.X_y_from_seq()
        else:
            self.X_pre, self.X_post, self.y = preprocess.X_y_from_seq()
        self.vocab_size = len(list(preprocess.vocab.keys()))
        self.seq_dict = preprocess.seq_dict
        self.vocab = preprocess.vocab
        self.min_len = preprocess.min_len
        self.truncate = preprocess.truncate

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.model_type == 'attention':
            return self.X[idx, :], self.y[idx, :]
        elif self.model_type == 'bilstm':
            return self.X_pre[idx, :], self.X_post[idx, :], self.y[idx, :]