# todo define semantics and grammaticality pipeline
import torch
from .lm_models import Transformer
from .utils import generate_mutations, aa_sequence, read_fasta, tokenize_and_pad, download_from_gcloud_bucket, mut_abbrev_to_seq
from collections import OrderedDict
import torch.nn.functional as F
import wandb
import ipdb


class CSCS_objective:
    def __init__(self, args, dataset, model=None, cscs_debug=False, model_type='attention'):
        torch.autograd.set_grad_enabled = False
        if model is None:
            model = load_empty_model(args, dataset)
        else:
            model.eval()
        self.model = model
        if args.job_dir is None:
            self.positions_file = args.POI_file
            self.wt_seqs_file = args.wt_seqs_file
        else:
            self.positions_file = download_from_gcloud_bucket(args.POI_file)
            self.wt_seqs_file = download_from_gcloud_bucket(args.wt_seqs_file)
        self.wt_seqs = read_fasta(self.wt_seqs_file)
        self.mut_seq_dict = generate_mutations(self.wt_seqs_file, self.positions_file)
        # escape_muts = ['S512Y', 'H526D', 'H526Y', 'N518D', 'P564L', 'K503N', 'S531Q', 'S512R', 'Q513E', 'S531Y', 'S509P', 'S574Y', 'R529S', 'S508P', 'I572M', 'R529G', 'S531M', 'H526S', 'S531F', 'I530W', 'I572D', 'S531W', 'Q513L', 'Q148H', 'D516G', 'S509R', 'H526E', 'H526G', 'H526P', 'S574F', 'R529L', 'S531L', 'S512P', 'L533P', 'H526F', 'R529K', 'Q148L', 'Q513N', 'L533H', 'I572S', 'Q513P', 'H526L', 'H526C', 'I572T', 'Q513R', 'D516A', 'D516V', 'D516P', 'Q513K', 'G534D', 'R529C', 'I572F', 'G570C', 'L511P', 'H526T', 'R529Q', 'H526R', 'H526Q', 'S522F', 'D516F', 'I572R', 'Q513F', 'L511Q', 'Q513H', 'S522Y', 'D516Y', 'H526N', 'R529H', 'L511R', 'P564R', 'T525R', 'S531C', 'S512F', 'D516N']
        # escape_seqs = [mut_abbrev_to_seq(mut, str(self.wt_seqs[0].seq)) for mut in escape_muts] + [str(self.wt_seqs[0].seq)]
        # self.mut_seq_dict[str(self.wt_seqs[0].seq)] = {k: v for k, v in self.mut_seq_dict[str(self.wt_seqs[0].seq)].items() if k in escape_seqs}
        # ipdb.set_trace()
        self.seq_dict = dataset.seq_dict
        self.dataset = dataset
        self.cscs_debug=cscs_debug
        self.model_type = model_type
        self.eval_batch_size = args.eval_batch_size
        if args.wandb:
            wandb.save(self.positions_file)
            wandb.save(self.wt_seqs_file)

        # with open(args.vocab_file, 'r') as j:
        #     self.vocab = json.loads(j.read())

    def compute_semantics(self):
        with torch.no_grad():
            tokenization_params = [self.dataset.vocab, self.dataset.max_len, self.dataset.truncate]
            for wt_seq in self.wt_seqs:
                wt_seq_tokens = tokenize_and_pad(self.model_type, [wt_seq], *tokenization_params)
                tokenized_wt_tensor = torch.Tensor(wt_seq_tokens[0][:-1])
                ipdb.set_trace()
                wt_embedding = self.model(tokenized_wt_tensor.unsqueeze(0).long())
                wt = str(wt_seq.seq)
                self.mut_seq_dict[wt][wt] = {}
                self.mut_seq_dict[wt][wt]['embedding'] = wt_embedding
                list_muts = list(self.mut_seq_dict[wt].keys())
                ipdb.set_trace()
                # mut_abbrev = self.mut_seq_dict[wt][list_muts[0]]['mut_abbrev']
                if self.cscs_debug:
                    list_muts = list_muts[:5]
                if self.model_type=='attention':
                    tokens = tokenize_and_pad(self.model_type, list_muts, *tokenization_params)
                    tokenized_tensor = torch.Tensor(tokens)[:, :-1]
                    if self.eval_batch_size is not None:
                        list_embeddings = []
                        for i in range((len(list_muts)//int(self.eval_batch_size)+1)):
                            start = i * int(self.eval_batch_size)
                            end = start + int(self.eval_batch_size)
                            subset_muts = tokenized_tensor[start:end, :]
                            subset_embeddings = self.model(subset_muts.long())
                            list_embeddings.append(subset_embeddings)
                        mut_embeddings = torch.cat(list_embeddings, dim=0)
                    else:
                        mut_embeddings = self.model(tokenized_tensor.long())
                elif self.model_type=='tape':
                    list_embeddings = []
                    for mut_seq in list_muts:
                        mut_embedding = self.model.forward(mut_seq).transpose(1, 2)
                        list_embeddings.append(mut_embedding)
                    mut_embeddings = torch.cat(list_embeddings, dim=0)
                elif self.model_type=='esm':
                    if self.eval_batch_size is not None:
                        list_embeddings = []
                        for i in range((len(list_muts)//int(self.eval_batch_size)+1)):
                            start = i * int(self.eval_batch_size)
                            end = start + int(self.eval_batch_size)
                            subset_muts = list_muts[start: end]
                            subset_input = [('Seq_{0}'.format(i), subset_muts[i]) for i in range(len(subset_muts))]
                            subset_embeddings = self.model.forward(subset_input)
                            list_embeddings.append(subset_embeddings)
                        mut_embeddings = torch.cat(list_embeddings, dim=0)
                    else:
                        inputs = [('Seq_{0}'.format(i), list_muts[i]) for i in range(len(list_muts))]
                        mut_embeddings = self.model.forward(inputs)
                else:
                    print("{0} model type not available".format(self.model_type))
                l1_norm = torch.sum(abs(wt_embedding-mut_embeddings), dim=[1, 2]).tolist()
                for mut, mut_embedding, l1_difference in zip(list_muts, mut_embeddings.tolist(), l1_norm):
                    self.mut_seq_dict[str(wt_seq.seq)][mut]['l1_semantic_diff'] = l1_difference
                    self.mut_seq_dict[str(wt_seq.seq)][mut]['embedding'] = mut_embedding
        return self

    def compute_grammar(self):
        for wt_seq in self.wt_seqs:
            list_muts = list(self.mut_seq_dict[str(wt_seq.seq)].keys())
            if self.cscs_debug:
                list_muts = list_muts[:5]
            for mut in list_muts:
                embedding = torch.Tensor(self.mut_seq_dict[str(wt_seq.seq)][mut]['embedding'])
                mutation_name = self.mut_seq_dict[str(wt_seq.seq)][mut]['mut_abbrev']
                pos = mutation_name[1:-1]
                aa = mutation_name[-1]
                softmax_embedding = F.softmax(embedding, dim=1)
                grammaticality = softmax_embedding[int(pos)-1, list(self.dataset.vocab.keys()).index(aa)]
                self.mut_seq_dict[str(wt_seq.seq)][mut]['grammaticality'] = grammaticality
            # state_dict = torch.load("net.pth")
            # self.mut_seq_dict = state_dict["mut_seq_dict"]  # Retrieving your stuff after loading
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
    checkpoint = torch.load(arg.state_dict_fname, map_location=torch.device('cpu') if not torch.cuda.is_available()
                            else torch.device("cuda: 0"))
    if not torch.cuda.is_available():
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")  # remove module.
            new_checkpoint[name] = v
    else:
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
    ipdb.set_trace()
#     cscs_computer = CSCS_objective(cscs_args, dataset, cscs_debug=True)
#     seq_dict = cscs_computer.compute_semantics()
#     seq_dict = cscs_computer.compute_grammar()

