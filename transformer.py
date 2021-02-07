import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.nn.functional import one_hot
# from torch.autograd import Variable
# import numpy as np
from torch.utils.data import Dataset
import ipdb
# from Bio.Seq import Seq
from Bio import SeqIO
import re
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import wandb
import uuid

def mask_fn(x, mask_diagonal=False):
    # ipdb.set_trace()
    b, h, w = x.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    mask = torch.zeros_like(x)
    mask[:, indices[0], indices[1]] = 1
    final_mask = (mask == 1) & (x == 0)
    x.masked_fill_(final_mask, float('-inf'))
    return x


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=1, mask=True):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(emb * heads, emb, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        keys = self.tokeys(x).contiguous().view(b, t, h, k)
        queries = self.toqueries(x).contiguous().view(b, t, h, k)
        values = self.tovalues(x).contiguous().view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        dot = torch.bmm(keys, queries.transpose(1, 2))  # (b*h, t, t)
        if self.mask:
            dot = mask_fn(dot)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).contiguous().view(b, t, h * k)
        # print(out.size())
        out = self.unifyheads(out)
        assert out.size() == (b, t, k)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb, hidden=4, heads=1, drop_prob=0.1, mask=True):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.feedforward = nn.Sequential(
            nn.Linear(emb, hidden * emb),
            nn.ReLU(),
            nn.Linear(hidden * emb, emb)
        )

        self.attention = SelfAttention(emb, heads=heads, mask=mask)

        self.LayerNorm1 = nn.LayerNorm(emb)
        self.LayerNorm2 = nn.LayerNorm(emb)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        attention = self.attention(x)
        x = self.LayerNorm1(x + attention)
        ff = self.feedforward(x)
        out = self.dropout(self.LayerNorm2(x + ff))
        return out


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden, embed_dim, heads, depth, seq_length, drop_prob=0.1, mask=True):
        super().__init__()
        self.token_embed = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(num_embeddings=seq_length, embedding_dim=embed_dim, padding_idx=0)

        blocks = []
        for layer in range(depth):
            blocks.append(TransformerBlock(emb=embed_dim, hidden=hidden, heads=heads, drop_prob=drop_prob, mask=mask))

        self.blocks = nn.Sequential(*blocks)
        self.to_prob = nn.Linear(embed_dim, vocab_size+1)

    def forward(self, x):
        embedding = self.token_embed(x) + self.pos_embed(x)
        b, t, k = embedding.size()
        out = self.blocks(embedding)
        probs = self.to_prob(out)
        # log_probs = F.log_softmax(probs)
        # return log_probs
        return probs


# class ENA_Data(Dataset):
#     def __init__(self, filename="ERR3929134_1.fastq"):
#         super().__init__()
#         records = list(SeqIO.parse("ERR3929134_1.fastq", "fastq"))
#
#         pass
#
#     def __len__(self):
#         return len(records)
#
#     def __getitem__(self):
#         nuc_sequence = records[0].seq
#         # todo – figure out how to get the protein sequence given the nucleotide sequence


class Preprocesser:
    def __init__(self, fname, condition, max_len=None, truncate=False, forbidden_aas=('X'), debug_mode=False):
        if forbidden_aas is None:
            forbidden_aas = ['X']
        root_dir = __file__.split('/')[:-1]
        root_dir = '/'.join(root_dir)
        fname = root_dir + '/{0}.fasta'.format(fname)
        self.debug_mode = debug_mode
        self.val = lambda x: 2000 if self.debug_mode else len(x) + 1
        self.truncate = truncate
        self.max_len = max_len
        self.fname = fname
        self.records = list(SeqIO.parse(fname, "fasta"))
        self.condition = condition  # tuple (species, value) or (identifier, value)
        self.metas = self.save_meta()
        self.forbidden_aas = list(forbidden_aas)
        self.seq_dict, self.num_seqs = self.collect_sequences()
        self.seqs = list(self.seq_dict.keys())
        if max_len is None:
            self.max_len = max([len(seq) for seq in self.seqs])
        amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
            'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
        ]
        relevant_aas = list(set(amino_acids)-set(self.forbidden_aas))
        self.vocab = {aa: idx + 1 for idx, aa in enumerate(relevant_aas)}

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
                cond3 = self.truncate == False and len(record.seq) > self.max_len
            if cond1 or cond2 or cond3:
                continue
            meta_info['seq_len'] = len(record.seq)
            seqs[record.seq] = []
            seqs[record.seq].append(meta_info)
        if self.condition is not None:
            print("After elimination of sequences with {0} forbidden amino acids and selection of sequences where {1} \
            is equal to {2}, {3} sequences are passed for training").format("".join([aa + ' ,' for aa in \
                                                                                     self.forbidden_aas]),
                                                                            self.condition[0], self.condition[1],
                                                                            len(seqs.keys()))
        else:
            print("After elimination of sequences with {0} forbidden amino acids. {1} sequences are passed for training"
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

    def pad(self, seq):
        if len(seq) > self.max_len and self.truncate == True:
            seq = seq[:self.max_len]
        elif len(seq) < self.max_len:
            seq.extend([0] * (self.max_len - len(seq)))
        return seq

    def tokenize_and_pad(self):
        padded_aas = []
        for seq in self.seqs:
            list_aa_indices = [self.vocab[char] for char in seq]
            padded_aa_indices = self.pad(list_aa_indices)
            padded_aas.append(padded_aa_indices)
        return padded_aas

    def X_y_from_seq(self):
        tokenized = self.tokenize_and_pad()
        tokenized_tensor = torch.Tensor(tokenized)
        # assert tokenized_tensor.size() == self.num_seqs, self.max_len
        X = tokenized_tensor[:, :-1]
        y = tokenized_tensor[:, 1:]
        # print(X.size(), y.size())
        # y = one_hot(tokenized_tensor[:, 1:].to(torch.int64), num_classes=len(self.vocab))
        return X, y

    def collate_fn(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        return batch


# todo – implement batch sampler if I want to keep all X, y's for a particular sequence together

class UniProt_Data(Dataset):
    def __init__(self, condition=None, max_len=None, truncate=False, forbidden_aas=('X'),
                 filename="uniprot_gpb_rpob", test=False):
        super().__init__()
        preprocess = Preprocesser(filename, condition, max_len=max_len, truncate=truncate,
                                  forbidden_aas=forbidden_aas, debug_mode=test)
        self.seqs = preprocess.seqs
        self.max_len = preprocess.max_len
        self.X, self.y = preprocess.X_y_from_seq()
        self.vocab_size = len(list(preprocess.vocab.keys()))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx, :]


def train(arg, data):
    if arg.wandb:
        unique_run_id = uuid.uuid1().hex[0:8]
        wandb.init(config={"Unique Run ID": unique_run_id}, project='Antimicrobial resistance', name=arg.name_run)
        wandb.config.update(arg)
    train_loader = DataLoader(data, batch_size=arg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    iterator = iter(train_loader)
    # print(sample[0].size(), sample[1].size())
    model = Transformer(dataset.vocab_size, hidden=arg.hidden, embed_dim=arg.embed_dim, heads=arg.heads, depth=arg.depth,
                        seq_length=dataset.max_len, drop_prob=0, mask=True)
    if torch.cuda.is_available():
        model.cuda()
    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    if arg.lr_scheduler is not None:
        if arg.lr_scheduler == 'lambda_lr':
            raise NotImplementedError
        if arg.lr_scheduler == 'plateau':
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=arg.patience, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=arg.weight, ignore_index=0)
    count = 0
    for e in range(arg.epochs):
        for t in range(len(train_loader)):
            model.train()
            X, y = next(iterator)
            out = model(X.long()).transpose(1, 2)
            training_loss = criterion(out, y.long())
            opt.zero_grad()
            training_loss.backward()
            opt.step()
            sch.step(training_loss)
            if t % arg.within_epoch_interval == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, training_loss.item()))
                count += 1
    pass

if __name__ == '__main__':
    dataset = UniProt_Data(filename="uniprot_gpb_rpob", max_len=2000, truncate=True, test=True)

    #todo build out argument parser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training set")
    parser.add_argument("--hidden", type=int, default=64, help="# of hidden layers for transformer block")
    parser.add_argument("--embed_dim", type=int, default=32, help="embedding dimension for transformer")
    parser.add_argument("--heads", type=int, default=4, help="# of heads for transformer")
    parser.add_argument("--depth", type=int, default=2, help='# of transformer block repeats')
    parser.add_argument("--drop_prob", type=float, default=0.1, help='Dropout for transformer blocks')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--model_type", choices=['attention'], default='attention', help="model to pretrain")
    parser.add_argument("--lr_scheduler", choices=['plateau', 'adaptive'], default='plateau')
    parser.add_argument("--within_epoch_interval", type=int, default=5, help="show update ever # of bactches")
    parser.add_argument("--patience", type=int, default=1, help="num epochs for lr scheduler to wait before decreasing")
    parser.add_argument("--wandb", action="store_true", help="log results using wandb")
    parser.add_argument('--name_run', type=str, help="name of run to save in wandb")

    args = parser.parse_args()

    train(args, dataset)





    # tens = torch.randint(1, 7, (10, 8))
    # print(tens.size())
    # tens = torch.cat((tens, torch.zeros([10, 3]).type(torch.LongTensor)), dim=1)
    # print(tens)
    # # hey = nn.Embedding(num_embeddings=7, embedding_dim=5, padding_idx=0)
    # # print(hey(tens))
    # trans = Transformer(vocab_size=6, hidden=3, embed_dim=5, heads=2, depth=3, seq_length=11)
    # print(trans(tens))
    # records = list(SeqIO.parse("ERR3929134_1.fastq", "fastq"))
    # ipdb.set_trace()
# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return torch.from_numpy(subsequent_mask) == 0
#
#
# def data_gen(V, batch, nbatches):
#     "Generate random data for a src-tgt copy task."
#     for i in range(nbatches):
#         data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
#         ipdb.set_trace()
#         data[:, 0] = 1
#         src = Variable(data, requires_grad=False)
#         tgt = Variable(data, requires_grad=False)
#         yield Batch(src, tgt, 0)
#
#
# class Batch:
#     "Object for holding a batch of data with mask during training."
#
#     def __init__(self, src, trg=None, pad=0):
#         self.src = src
#         self.src_mask = (src != pad).unsqueeze(-2)
#         if trg is not None:
#             self.trg = trg[:, :-1]
#             self.trg_y = trg[:, 1:]
#             self.trg_mask = \
#                 self.make_std_mask(self.trg, pad)
#             self.ntokens = (self.trg_y != pad).data.sum()
#
#     @staticmethod
#     def make_std_mask(tgt, pad):
#         "Create a mask to hide padding and future words."
#         tgt_mask = (tgt != pad).unsqueeze(-2)
#         tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
#         return tgt_mask
