from .utils import read_fasta, tokenize_and_pad, download_from_gcloud_bucket
from .cscs import load_empty_model
import torch
import wandb
import scanpy as sc
from scanpy import AnnData
import numpy as np
import ipdb
from os import path


def analyze_embeddings(args, dataset, model, wt_fname, uniprot_fnames, embedding_fname=None, namespace='rpoB'):
    wt_seqs = read_fasta(wt_fname)

    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    if path.exists(embedding_fname):
        seq_dict = torch.load(embedding_fname, map_location=device)
    else:
        if args.job_dir is not None:
            embedding_fname = download_from_gcloud_bucket(embedding_fname)
            seq_dict = torch.load(embedding_fname, map_location=device)
        else:
            seq_dict = {}

    model = model if model is not None else load_empty_model(args, dataset)
    device = "cpu"
    if torch.cuda.is_available():
        model.cuda()
        device = "cuda"

    X, obs = [], {}
    obs['seq'] = []
    for seq, uniprot_fname in zip(wt_seqs, uniprot_fnames):
        uniprot_seqs = read_fasta(uniprot_fname)
        uniprot_seqs = list(set([str(seq.seq) for seq in uniprot_seqs if args.max_len > len(str(seq.seq)) > args.min_len
                                 and 'X' not in seq]))
        wt = str(seq.seq)
        if wt not in seq_dict:
            seq_dict[wt] = {}

        wt_seq_dict = seq_dict[wt]
        seq_tokens = tokenize_and_pad(args.model_type, uniprot_seqs, dataset.vocab, args.max_len, args.truncate)
        seq_tokens = torch.Tensor(seq_tokens).to(device)
        embedding = torch.mean(model(seq_tokens[:, :-1].long(), repr_layers=[args.depth - 1]), dim=-2)[0]
        for i, uniprot_seq in enumerate(uniprot_seqs):
            if uniprot_seq not in wt_seq_dict:
                wt_seq_dict[uniprot_seq] = {}
            wt_seq_dict[uniprot_seq]['mean_embedding'] = embedding[i]
        X.append(embedding.numpy())
        obs['seq'].extend(uniprot_seqs)
        X = np.array(X)
        adata = AnnData(X)
        for key in obs:
            adata.obs[key] = obs[key]

        sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)

        sc.set_figure_params(dpi_save=500)
        plot_umap(adata, namespace)
        torch.save(seq_dict, embedding_fname)
        if args.wandb:
            wandb.save('{}_louvain.png'.format(namespace))
            wandb.save(embedding_fname)
    return embedding_fname
        # interpret_clusters(adata)


def plot_umap(adata, namespace):
    sc.tl.umap(adata, min_dist=1.)
    sc.pl.umap(adata, color='louvain', save='{}_louvain.png'.format(namespace))
    # sc.pl.umap(adata, color='subtype', save='{}_subtype.png'.format(namespace))


def interpret_clusters():
    pass


if __name__ == '__main__':
    analyze_embeddings("saved_models/third_trial_47.pth", "escape_validation/anchor_seqs.fasta",
                       ["uniprot_gpb_rpob.fasta"])
