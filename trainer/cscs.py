# todo define semantics and grammaticality pipeline
import torch
from .lm_models import Transformer
from .utils import generate_mutations, aa_sequence, read_fasta, tokenize_and_pad, download_from_gcloud_bucket
from collections import OrderedDict
import torch.nn.functional as F
import wandb
import ipdb


class CSCS_objective:
    def __init__(self, args, dataset, model=None, cscs_debug=False):
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
        self.seq_dict = dataset.seq_dict
        self.dataset = dataset
        self.cscs_debug=cscs_debug
        self.model_type = args.model_type
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
                tokenized_wt_tensor = torch.Tensor(wt_seq_tokens[0])
                wt_embedding = self.model(tokenized_wt_tensor.unsqueeze(0).long())
                list_muts = list(self.mut_seq_dict[str(wt_seq.seq)].keys())
                if self.cscs_debug:
                    list_muts = list_muts[:5]
                tokens = tokenize_and_pad(self.model_type, list_muts, *tokenization_params)
                tokenized_tensor = torch.Tensor(tokens)
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
                l1_norm = abs(torch.sum(wt_embedding-mut_embeddings, dim=[1, 2])).tolist()
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

    dataset = UniProt_Data(filename="escape_validation/anchor_seqs", max_len=cscs_args.max_len, truncate=cscs_args.truncate,
                           test=cscs_args.cscs_debug,
                           job_dir=cscs_args.job_dir)
    model = load_empty_model(cscs_args, dataset)
    ipdb.set_trace()
#     cscs_computer = CSCS_objective(cscs_args, dataset, cscs_debug=True)
#     seq_dict = cscs_computer.compute_semantics()
#     seq_dict = cscs_computer.compute_grammar()

