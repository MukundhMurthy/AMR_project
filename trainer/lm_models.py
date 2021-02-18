import torch.nn as nn
from trainer.utils import mask_fn
import torch
from torch.nn import functional as F
# from tape import ProteinBertModel, TAPETokenizer


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

        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))

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
        self.token_embed = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(num_embeddings=seq_length, embedding_dim=embed_dim, padding_idx=0)

        blocks = []
        for layer in range(depth):
            blocks.append(TransformerBlock(emb=embed_dim, hidden=hidden, heads=heads, drop_prob=drop_prob, mask=mask))

        self.blocks = nn.Sequential(*blocks)
        self.to_prob = nn.Linear(embed_dim, vocab_size + 2)

    def forward(self, x):
        embedding = self.token_embed(x) + self.pos_embed(x)
        b, t, k = embedding.size()
        out = self.blocks(embedding)
        probs = self.to_prob(out)
        # log_probs = F.log_softmax(probs)
        # return log_probs
        return probs


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_dim, seq_length, num_layers, bidirectional=True, drop_prob=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(seq_length, hidden_size, num_layers, bias=True, batch_first=True, dropout=0,
                              bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.dnn = nn.Linear(hidden_size * bidirectional, vocab_size+2)

    def forward(self, x_pre, x_post):
        x_pre_embedding = self.token_embed(x_pre)
        x_post_embedding = self.token_embed(x_post)
        lstm_pre = self.bilstm(x_pre_embedding)
        lstm_post = self.bilistm(x_post_embedding)
        concat = torch.cat((lstm_pre, lstm_post), dim=-1)
        probs_vec = F.softmax(self.dnn(concat)[-1], dim=-1)
        return probs_vec


# class fb_esm:
#     def __init__(self):
#         self.model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
#         self.batch_converter = alphabet.get_batch_converter()
#
#     def forward(self, data):
#         with torch.no_grad():
#             batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
#             results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
#         token_representations = results["representations"][33]
#         return token_representations
#

# class Tape_model:
#     def __init__(self):
#         self.model = ProteinBertModel.from_pretrained('bert-base')
#         self.tokenizer = TAPETokenizer(vocab='iupac')
#
#     def forward(self, sequence):
#         token_ids = torch.tensor([self.tokenizer.encode(sequence)])
#         output = self.model(token_ids)
#         sequence_output = output[0]
#         return sequence_output






