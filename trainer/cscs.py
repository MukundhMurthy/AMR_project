# todo define semantics and grammaticality pipeline
import torch
from .lm_models import Transformer
from .utils import generate_mutations, aa_sequence, read_fasta, tokenize_and_pad, download_from_gcloud_bucket, mut_abbrev_to_seq
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import wandb
import ipdb
import numpy as np
from tqdm import tqdm


class CSCS_objective:
    def __init__(self, args, dataset=None, model=None, cscs_debug=False, model_type='attention'):
        torch.autograd.set_grad_enabled = False
        if model is None:
            model = load_empty_model(args, dataset)
        else:
            if model_type=='attention':
                model.eval()
        self.model = model
        if args.job_dir is None:
            self.positions_file = args.POI_file
            self.wt_seqs_file = args.wt_seqs_file
        else:
            self.positions_file = download_from_gcloud_bucket(args.POI_file)
            self.wt_seqs_file = download_from_gcloud_bucket(args.wt_seqs_file)
        self.wt_seqs = read_fasta(self.wt_seqs_file)
        # if not (args.train and not args.calc_metrics):
        #     ipdb.set_trace()
        #     self.mut_seq_dict = generate_mutations(self.wt_seqs_file, self.positions_file) if state_dict_fname is None else torch.load(state_dict_fname, map_location=device)
        # else:
        # self.mut_seq_dict = torch.load("saved_models/compressed_comb.pth", map_location=torch.device("cpu"))
        self.mut_seq_dict = generate_mutations(self.wt_seqs_file, self.positions_file)
        # escape_muts = ['S512Y', 'H526D', 'H526Y', 'N518D', 'P564L', 'K503N', 'S531Q', 'S512R', 'Q513E', 'S531Y', 'S509P', 'S574Y', 'R529S', 'S508P', 'I572M', 'R529G', 'S531M', 'H526S', 'S531F', 'I530W', 'I572D', 'S531W', 'Q513L', 'Q148H', 'D516G', 'S509R', 'H526E', 'H526G', 'H526P', 'S574F', 'R529L', 'S531L', 'S512P', 'L533P', 'H526F', 'R529K', 'Q148L', 'Q513N', 'L533H', 'I572S', 'Q513P', 'H526L', 'H526C', 'I572T', 'Q513R', 'D516A', 'D516V', 'D516P', 'Q513K', 'G534D', 'R529C', 'I572F', 'G570C', 'L511P', 'H526T', 'R529Q', 'H526R', 'H526Q', 'S522F', 'D516F', 'I572R', 'Q513F', 'L511Q', 'Q513H', 'S522Y', 'D516Y', 'H526N', 'R529H', 'L511R', 'P564R', 'T525R', 'S531C', 'S512F', 'D516N']
        # escape_seqs = [mut_abbrev_to_seq(mut, str(self.wt_seqs[0].seq)) for mut in escape_muts] + [str(self.wt_seqs[0].seq)]
        # self.mut_seq_dict[str(self.wt_seqs[0].seq)] = {k: v for k, v in self.mut_seq_dict[str(self.wt_seqs[0].seq)].items() if k in escape_seqs}
        self.dataset = dataset
        self.cscs_debug=cscs_debug
        self.model_type = model_type
        self.eval_batch_size = args.eval_batch_size
        self.depth = args.depth
        if args.wandb:
            wandb.save(self.positions_file)
            wandb.save(self.wt_seqs_file)

        # with open(args.vocab_file, 'r') as j:
        #     self.vocab = json.loads(j.read())

    def compute_semantics(self):
        with torch.no_grad():
            for wt_seq in self.wt_seqs:
                wt = str(wt_seq.seq)
                if self.model_type == 'attention':
                    tokenization_params = [self.dataset.vocab, self.dataset.max_len, self.dataset.truncate]
                    wt_seq_tokens = tokenize_and_pad(self.model_type, [wt_seq], *tokenization_params)
                    tokenized_wt_tensor = torch.Tensor(wt_seq_tokens[0][:-1])
                    wt_embedding = self.model(tokenized_wt_tensor.unsqueeze(0).long(), repr_layers=[self.depth-1])[0]
                else:
                    wt_embedding = self.model.forward([('wt_seq', "".join(list(wt)[:1022]))])
                self.mut_seq_dict[wt][wt] = {}
                self.mut_seq_dict[wt][wt]['embedding'] = wt_embedding
                self.mut_seq_dict[wt][wt]['mut_abbrev'] = 'M1M'
                self.mut_seq_dict[wt][wt]['cluster'] = 'None'

                list_muts = list(self.mut_seq_dict[wt].keys())
                if self.cscs_debug:
                    list_muts = list_muts[:5]
                if self.model_type == 'attention':
                    tokens = tokenize_and_pad(self.model_type, list_muts, *tokenization_params)
                    tokenized_tensor = torch.Tensor(tokens)[:, :-1]
                    if self.eval_batch_size is not None:
                        list_embeddings = []
                        list_prob_embeddings = []
                        for i in tqdm(range((len(list_muts)//int(self.eval_batch_size)+1))):
                            start = i * int(self.eval_batch_size)
                            end = start + int(self.eval_batch_size)
                            subset_muts = tokenized_tensor[start:end, :]
                            subset_embeddings = self.model(subset_muts.long(), repr_layers=[self.depth-1])[0]
                            subset_prob_embeddings = self.model(subset_muts.long())
                            list_embeddings.append(subset_embeddings)
                            list_prob_embeddings.append(subset_prob_embeddings)
                        mut_embeddings = torch.cat(list_embeddings, dim=0)
                        mut_prob_embeddings = torch.cat(list_prob_embeddings, dim=0)
                    else:
                        mut_embeddings = self.model(tokenized_tensor.long(), repr_layers=[self.depth-1])[0]
                elif self.model_type=='tape':
                    list_embeddings = []
                    for mut_seq in list_muts:
                        mut_embedding = self.model.forward(mut_seq).transpose(1, 2)
                        list_embeddings.append(mut_embedding)
                    mut_embeddings = torch.cat(list_embeddings, dim=0)
                elif self.model_type=='esm':
                    if self.eval_batch_size is not None:
                        list_embeddings = []
                        for i in tqdm(range((len(list_muts)//int(self.eval_batch_size)+1))):
                            start = i * int(self.eval_batch_size)
                            end = start + int(self.eval_batch_size)
                            subset_muts = list_muts[start: end]
                            subset_input = [('Seq_{0}'.format(i), "".join(list(subset_muts[i])[:1022])) for i in range(len(subset_muts))]
                            subset_embeddings = self.model.forward(subset_input)
                            list_embeddings.append(subset_embeddings)
                        mut_embeddings = torch.cat(list_embeddings, dim=0)
                    else:
                        inputs = [('Seq_{0}'.format(i), list_muts[i]) for i in range(len(list_muts))]
                        mut_embeddings = self.model.forward(inputs)
                else:
                    print("{0} model type not available".format(self.model_type))
                l1_norm = torch.sum(abs(wt_embedding-mut_embeddings), dim=[1, 2]).tolist()
                for mut, l1_difference, mut_prob_embedding in zip(list_muts, l1_norm, mut_prob_embeddings.tolist()):
                    self.mut_seq_dict[str(wt_seq.seq)][mut]['l1_semantic_diff'] = l1_difference
                    self.mut_seq_dict[str(wt_seq.seq)][mut]['embedding'] = mut_prob_embedding
        return self

    def compute_grammar(self):
        for wt_seq in self.wt_seqs:
            list_muts = list(self.mut_seq_dict[str(wt_seq.seq)].keys())
            mutation_names = [self.mut_seq_dict[str(wt_seq.seq)][mut]['mut_abbrev'] for mut in list_muts]
            posit = [mutation_name[1:-1] for mutation_name in mutation_names]
            list_aa = [mutation[-1] for mutation in mutation_names]
            if self.cscs_debug:
                list_muts = list_muts[:5]
            if self.model_type == 'attention':
                for mut, pos, aa in zip(list_muts, posit, list_aa):
                    embedding = torch.Tensor(self.mut_seq_dict[str(wt_seq.seq)][mut]['embedding'])
                    softmax_embedding = F.softmax(embedding, dim=1)
                    grammaticality = softmax_embedding[int(pos)-1, list(self.dataset.vocab.keys()).index(aa)]
                    self.mut_seq_dict[str(wt_seq.seq)][mut]['grammaticality'] = grammaticality
            elif self.model_type == 'esm':
                grammar = self.model.forward_grammar(self.eval_batch_size, list_muts, list_aa, posit)
                for mut, grammar in zip(list_muts, grammar):
                    self.mut_seq_dict[str(wt_seq.seq)][mut]['grammaticality'] = grammar
            # elif self.model_type == 'tape':

        return self.mut_seq_dict

    def compute_grammar_bilstm(self):
        raise NotImplementedError

    @staticmethod
    def get_aa_sequence(seq):
        return aa_sequence(seq)


def load_empty_model(arg, dataset):
    empty_model = Transformer(dataset.vocab_size, hidden=arg.hidden, embed_dim=arg.embed_dim, heads=arg.heads,
                              depth=arg.depth,
                              seq_length=dataset.max_len, drop_prob=arg.drop_prob, mask=True)
    state_dict_fname = download_from_gcloud_bucket(arg.state_dict_fname) if arg.job_dir else arg.state_dict_fname
    checkpoint = torch.load(state_dict_fname, map_location=torch.device('cpu') if not torch.cuda.is_available()
                            else torch.device("cuda"))
    if not torch.cuda.is_available():
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")  # remove module.
            new_checkpoint[name] = v
    else:
        print('calling DDP on instantiated model')
        #torch.distributed.init_process_group(
        #        backend='nccl',
        #        init_method='env://'
        #)
        n_devices = torch.cuda.device_count()
        device_ids = np.arange(0, n_devices).tolist()
        empty_model = torch.nn.parallel.DataParallel(empty_model, device_ids=device_ids)
        new_checkpoint = checkpoint


    empty_model.load_state_dict(new_checkpoint, strict=False)
    model = empty_model
    model.eval()
    return model



if __name__ == "__main__":
    from .main import cscs_parser
    from .main import train_parser
    from .preprocess import UniProt_Data
    print('goo')
    parser = train_parser()
    cscs_parser = cscs_parser(parser)
    cscs_args = cscs_parser.parse_args()

    dataset = UniProt_Data(filename="uniprot_gpb_rpob", min_len=cscs_args.min_len, max_len=cscs_args.max_len, truncate=cscs_args.truncate,
                           test=cscs_args.cscs_debug,
                           job_dir=cscs_args.job_dir)
    cscs_computer = CSCS_objective(cscs_args, dataset, model=None, cscs_debug=cscs_args.cscs_debug)
    cscs_computer = cscs_computer.compute_semantics()
#     cscs_computer = CSCS_objective(cscs_args, dataset, cscs_debug=True)
#     seq_dict = cscs_computer.compute_semantics()
#     seq_dict = cscs_computer.compute_grammar()

