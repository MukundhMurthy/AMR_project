import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.autograd import Variable
# import numpy as np
from torch.utils.data import Dataset
import ipdb
# from Bio.Seq import Seq
from Bio import SeqIO
import re


def mask(x, mask_diagonal=False):
    ipdb.set_trace()
    b, h, w = x.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    x[:, indices[0], indices[1]] = float('-inf')
    x.masked_fill_(x == 0, float('-inf'))
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

        self.unifyheads = nn.Linear(emb*heads, emb, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        keys = self.tokeys(x).contiguous().view(b, t, h, k)
        queries = self.toqueries(x).contiguous().view(b, t, h, k)
        values = self.tovalues(x).contiguous().view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b*h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b*h, t, k)
        values = values.transpose(1, 2).contiguous().view(b*h, t, k)

        dot = torch.bmm(keys, queries.transpose(1, 2))  #(b*h, t, t)
        if self.mask:
            dot = mask(dot)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).contiguous().view(b, t, h*k)
        print(out.size())
        out = self.unifyheads(out)
        assert out.size() == (b, t, k)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb, hidden=4, heads=1, drop_prob=0.1, mask=True):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask=mask

        self.feedforward = nn.Sequential(
            nn.Linear(emb, hidden*emb),
            nn.ReLU(),
            nn.Linear(hidden*emb, emb)
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
        self.to_prob = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedding = self.token_embed(x) + self.pos_embed(x)
        b, t, k = embedding.size()
        out = self.blocks(embedding)
        probs = self.to_prob(out)
        log_probs = F.log_softmax(probs)
        return log_probs


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
    def __init__(self, fname, condition, max_len=None, truncate=False):
        root_dir = __file__.split('/')[:-1]
        root_dir = '/'.join(root_dir)
        fname = root_dir + '/{0}.fasta'.format(fname)
        self.fname = fname
        self.records = list(SeqIO.parse(fname, "fasta"))
        self.condition = condition # tuple (species, value) or (identifier, value)

        seq_dict, self.num_seqs = self.collect_sequences()
        self.seqs = list(seq_dict.keys())
        self.max_len = max([len(seq) for seq in self.seqs]) if max_len is None else max_len
        if max_len is not None:
            self.truncate = truncate
        self.metas = self.save_meta()
        amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
            'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
        ]
        self.vocab = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}

    def collect_sequences(self):
        seqs={}
        for record in self.records:
            cond1 = "X" in record.seq
            access = self.get_access(record.description)
            meta_info = self.metas[access]
            cond2 = self.condition is not None and meta_info[self.condition[0]]!=self.condition[1]
            cond3 = self.truncate==False and len(record.seq)>self.max_len
            if cond1 or cond2 or cond3:
                continue
            meta_info['seq_len'] = len(record.seq)
            seqs[record.seq].append(meta_info)
        print("After elimination of sequences with X amino acids and selection of sequences where {0} \
        is equal to {1}, {2} sequences are passed for training").format(self.condition[0], self.condition[1], \
                                                                        len(seqs.keys()))
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
                accession = self.get_access(full_line)
                metas[accession] = {
                    'protein_entry': re.findall('(?<=....\|).*?(?=\s)', full_line),
                    'gene_entry': re.findall('(?<=\s).*?(?=\sOS)', full_line),
                    'organism_name': re.findall('(?<=OS\=).*?(?=\sOX\=)', full_line),
                    'organism_identifier': re.findall('(?<=OX\=).*?(?=\sGN\=)', full_line),
                    'gene_name': re.findall('(?<=GN\=).*?(?=\sPE\=)', full_line),
                    'protein_existence': re.findall('(?<=PE\=)\d(?=\sSV\=)', full_line),
                    'sequence_version': re.findall('(?<=SV\=).*$', full_line)
                }
        return metas

    def pad(self, seq):
        if len(seq)>self.max_len and self.truncate==True:
            seq = seq[:self.max_len]
        elif len(seq)<self.max_len:
            seq = seq.extend([0] * (self.max_len - len(seq)))
        return seq

    def tokenize(self):
        for seq in self.seqs:
            list_aa_indices = [self.vocab[char] for char in seq]
        padded_aa_indices = self.pad(list_aa_indices)
        return padded_aa_indices


    def collate_fn(batch):

        return batch


class UniProt_Data(Dataset):
    def __init__(self, condition, filename="uniprot_gpb_rpob"):
        super().__init__()
        preprocess = Preprocesser(filename, condition)
        self.seqs = preprocess.seqs
        self.maxlen = preprocess.maxlen

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()









if __name__ == '__main__':
    UniProt_Data("uniprot_gpb_rpob")
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