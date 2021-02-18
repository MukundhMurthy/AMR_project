from .utils import read_fasta, mut_abbrev_to_seq, generate_vocab, calc_bedroc
from .load_data import load_rpob_data
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
import numpy as np

# genes must be in order A, B, C
class Metrics:
    def __init__(self, embeddings_file, wt_seq_file, file_column_dict, beta=1, gene='rpoB', alpha=20):

        self.vocab = generate_vocab(forbidden_aa='X')
        self.gene = gene
        self.results_dict = {
            'rpoA': {
              'primary': {}
            },
            'rpoB': {
                'primary': {},
                'secondary': {
                    'single': {},
                    'combinatoric': {},
                }
            },
            'rpoC': {
                'primary': {},
            }
        }

        self.wt_seq_file = wt_seq_file
        self.wt_seqs = read_fasta(self.wt_seq_file)
        anchor_ids_dict = {
            'rpoB': 'tr|A0A0F6B9Y4|A0A0F6B9Y4_SALT1'
        }

        func_dict = {
            'rpoA': self.load_rpoa,
            'rpoB': self.load_rpob,
            'rpoC': self.load_rpoc
        }
        self.embedding_file = embeddings_file
        self.beta = beta
        self.weight_dict = torch.load(self.embedding_file)
        ids = [seq.id for seq in self.wt_seqs]
        seqs = [str(seq.seq) for seq in self.wt_seqs]
        id_to_seq = {id: seq for id, seq in zip(ids, seqs)}
        self.gene_wt_dict = {gene: id_to_seq[anchor_ids_dict[gene]] for gene in list(anchor_ids_dict.keys())}
        self.file_column_dictionary = file_column_dict
        # dictionary which shows the columns to compute metrics from files
        # format is {{filename: {mutation_column_name: name, eval_column_names: name}}}
        for anchor in list(anchor_ids_dict.keys()):
            self.df_dict = func_dict[anchor]()

        self.primary_fnames = [k for k, v in self.df_dict.items() if 'primary' in k]
        self.comp_fnames = [k for k, v in self.df_dict.items() if 'comp' in k]
        self.alpha = 20

    def load_rpoa(self):
        raise NotImplementedError

    def embedding2grammar(self, gene_mut_seq_dict, seq, mut_name):
        embedding = gene_mut_seq_dict[seq]['embedding']
        softmax_embedding = F.softmax(embedding, dim=1)
        pos = int(mut_name[1:-1])
        aa = mut_name[-1]
        grammar = softmax_embedding[int(pos) - 1, list(self.vocab.keys()).index(aa)]
        return grammar

    def fname2cscs(self, fname, df_dict, gene_mut_seq_dict, wt, combinatoric=False, ori_wt=None, primary_mut=None):
        mut_column = self.file_column_dictionary[fname]['mutation_column_name']
        mut_list = df_dict[fname]['df'][mut_column].tolist()
        semantics_list, grammar_list = [], []
        for mut in mut_list:
            mut_seq = mut_abbrev_to_seq(mut, wt)
            if not combinatoric:
                grammar = self.embedding2grammar(gene_mut_seq_dict, wt, mut)
                grammar_list.append(grammar)
                if wt not in gene_mut_seq_dict:
                    semantics = gene_mut_seq_dict[mut_seq]['l1_semantic_diff']
                else:
                    semantics = gene_mut_seq_dict[mut_seq]['l1_semantic_diff'] - \
                                gene_mut_seq_dict[wt]['l1_semantic_diff']
                semantics_list.append(semantics)
            else:
                primary_grammar = self.embedding2grammar(gene_mut_seq_dict, ori_wt, primary_mut)
                secondary_grammar = self.embedding2grammar(gene_mut_seq_dict, wt, mut)
                grammar = primary_grammar * secondary_grammar
                semantics = gene_mut_seq_dict[mut_seq]['l1_semantic_diff']
                grammar_list.append(grammar)
                semantics_list.append(semantics)
        semantic_ranks = rankdata(semantics_list)
        grammar_ranks = rankdata(grammar_list)
        cscs = semantic_ranks + self.beta * grammar_ranks
        mut_seqlist = [mut_abbrev_to_seq(mut, wt) for mut in mut_list]
        if not combinatoric:
            for cscs_score, grammar_score, semantic_score, mutation in zip(cscs, grammar_list, semantics_list, mut_seqlist):
                gene_mut_seq_dict[mutation]['cscs_ranked'] = cscs_score
                gene_mut_seq_dict[mutation]['grammaticality'] = grammar_score
                gene_mut_seq_dict[mutation]['l1_semantic_diff'] = semantic_score
        return cscs, gene_mut_seq_dict


    def load_rpob(self):
        wt = self.gene_wt_dict['rpoB']
        rpob_mut_seq_dict = self.weight_dict[wt]

        for fname in self.primary_fnames:
            cscs, rpob_mut_seq_dict = self.fname2cscs(fname, self.df_dict, rpob_mut_seq_dict, wt)
            eval_columns = self.file_column_dictionary[fname]['eval_column_names']

            fname_results = {eval_column: list(spearmanr(self.file_column_dictionary[fname][eval_column], cscs))
                             for eval_column in eval_columns}
            self.results_dict['primary'][fname] = fname_results

        for fname in self.comp_fnames:
            primary_mutation = self.df_dict[fname]['baseline_mutation']
            primary_mut_seq = mut_abbrev_to_seq(primary_mutation, wt)
            single_cscs, rpob_mut_seq_dict = self.fname2cscs(fname, self.df_dict, rpob_mut_seq_dict, primary_mut_seq)
            comb_cscs, rpob_mut_seq_dict = self.fname2cscs(fname, self.df_dict, rpob_mut_seq_dict, primary_mut_seq, combinatoric=True,
                                        ori_wt=wt, primary_mut=primary_mutation)
            eval_columns = self.file_column_dictionary[fname]['eval_column_names']
            single_results = {eval_column: list(spearmanr(self.file_column_dictionary[fname][eval_column], single_cscs))
                              for eval_column in eval_columns}
            comb_results = {eval_column: list(spearmanr(self.file_column_dictionary[fname][eval_column], comb_cscs))
                                    for eval_column in eval_columns}
            self.results_dict['secondary']['single'][fname] = single_results
            self.results_dict['secondary']['combinatoric'][fname] = comb_results

        self.weight_dict[wt] = rpob_mut_seq_dict
        torch.save(self.weight_dict, self.wt_seq_file)
        return self.results_dict, rpob_mut_seq_dict

    def gather_all_escape_muts(self):
        mutation_list = []
        for fname in self.primary_fnames:
            mut_column_name = self.file_column_dictionary[fname]['mutation_column_name']
            mutation_list.append(self.df_dict[fname]['df'][mut_column_name].tolist())
        mutations = itertools.chain(mutation_list)
        return mutations

    def load_rpoc(self):
        raise NotImplementedError

    def auc(self):
        raise NotImplementedError

    def escape_metrics(self, mut_seq_dict, semantic_scaling, grammatical_scaling, cscs_scaling):
        func_dict = {
            'min_max': MinMaxScaler,
            'standard': StandardScaler,
            'robust': RobustScaler
        }
        list_primary_mutations = self.gather_all_escape_muts()

        grammar_list, semantic_list = [], []
        idx_list = []
        for idx, (mut, _) in enumerate(mut_seq_dict):
            semantics = mut_seq_dict[mut]['l1_semantic_diff']
            mut_name = mut_seq_dict[mut]['mut_abbrev']
            if mut_name in list_primary_mutations:
                idx_list.append(idx)

            grammar = self.embedding2grammar(mut_seq_dict, mut, mut_name)
            grammar_list.append(grammar)
            semantic_list.append(semantics)


        fig, ax = plt.subplots(1, 2)
        ax[0][0].plot(grammar_list)
        ax[0][0].set_title('grammaticality distribution')
        ax[0][1].plot(semantic_list)
        ax[0][1].set_title('semantics distribution')

        fig.savefig('cscs_distributions.png')

        grammar_scaled = func_dict[grammatical_scaling]().fit_transform(grammar_list)
        semantics_scaled = func_dict[semantic_scaling]().fit_transform(semantic_list)

        cscs = grammar_scaled + self.beta * semantics_scaled
        cscs_scaled = np.array(func_dict[cscs_scaling]().fit_transform(cscs))

        y_true = np.zeros(len(grammar_list))
        y_true[idx_list] = 1

        cscs_aps = average_precision_score(y_true, cscs_scaled)
        semantic_aps = average_precision_score(y_true, semantics_scaled)
        grammar_aps = average_precision_score(y_true, grammatical_scaling)

        list_arrays = [np.tranpose(np.array([y_true, measure]) for measure in [cscs_scaled,
                                                                               semantics_scaled, grammar_scaled])]
        sorted_arrays = [data_array[np.argsort(data_array[:, 1])] for data_array in list_arrays]

        cscs_bedroc, semantic_bedroc, grammar_bedroc = [calc_bedroc(array[:, 0], self.alpha) for array in sorted_arrays]

        self.results_dict['primary']['overall_escape_id'] = {
            'cscs BEDROC': cscs_bedroc,
            'semantic BEDROC': semantic_bedroc,
            'grammar BEDROC': grammar_bedroc,
            'cscs APS': cscs_aps,
            'semantic APS': semantic_aps,
            'grammar APS': grammar_aps
        }
        return cscs_aps, semantic_aps, grammar_aps



