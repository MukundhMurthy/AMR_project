import torch.nn as nn
from trainer.utils import mask_fn
import torch
from torch.nn import functional as F
import ipdb
from tape import ProteinBertModel, TAPETokenizer


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

    def forward(self, x, repr_layers=None):
        embedding = self.token_embed(x) + self.pos_embed(x)
        b, t, k = embedding.size()

        if repr_layers is not None:
            assert max(repr_layers) <= len(self.blocks) - 1

        repr_layer_results = []

        for i, block in enumerate(self.blocks):
            embedding = block(embedding)

            if repr_layers is not None:
                if i in repr_layers:
                    repr_layer_results.append(embedding)

        if repr_layers is None:
            out = self.to_prob(embedding)
        else:
            out = repr_layer_results

        # log_probs = F.log_softmax(probs)
        # return log_probs
        return out


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_dim, num_layers, bidirectional=True, drop_prob=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, num_layers, bias=True, batch_first=True, dropout=0,
                              bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.dnn = nn.Linear(2 * hidden_size * num_directions, vocab_size+2)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x_pre, x_post):
        assert max(x_pre[0].tolist()) == self.vocab_size+1
        x_pre_embedding = self.token_embed(x_pre)
        x_post_embedding = self.token_embed(x_post)
        lstm_pre = self.bilstm(x_pre_embedding)[0]
        lstm_post = self.bilstm(x_post_embedding)[0]
        concat = torch.cat((lstm_pre, lstm_post), dim=-1)
        out = self.dropout(self.dnn(concat)[:, -1, :])
        probs_vec = F.softmax(out, dim=-1)
        return probs_vec
        # probs_vec = F.softmax(self.dnn(concat)[-1], dim=-1)
        # return probs_vec


class fb_esm:
    def __init__(self):
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.lm_head = self.model.lm_head
        self.vocab = self.alphabet.get_idx

    def forward(self, data):
        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        return token_representations

    def forward_grammar(self, eval_batch_size, batch_data, list_aa, posit):
        grammar_list = []
        for i in range((len(list_aa) // int(eval_batch_size) + 1)):
            start = i * int(eval_batch_size)
            end = start + int(eval_batch_size)
            subset_muts = batch_data[start: end]
            subset_input = [('Seq_{0}'.format(i), subset_muts[i]) for i in range(len(subset_muts))]
            subset_embeddings = self.model.forward(subset_input)
            probs = self.model.lm_head(subset_embeddings)
            softmax_probs = F.softmax(probs)
            idxs = [self.vocab(aa) for aa in list_aa[start:end]]
            count = torch.arange(eval_batch_size)
            positions = posit[start:end]
            grammar = softmax_probs[count, positions, idxs]
            grammar_list.extend(grammar)
        return grammar_list


class Tape_model:
    def __init__(self):
        self.model = ProteinBertModel.from_pretrained('bert-base')
        self.tokenizer = TAPETokenizer(vocab='iupac')

    def forward(self, sequence):
        token_ids = torch.tensor([self.tokenizer.encode(sequence)])
        output = self.model(token_ids)
        sequence_output = output[0]
        return sequence_output

    def forward_grammar(self):


        pass

