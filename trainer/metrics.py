from .utils import read_fasta, mut_abbrev_to_seq, generate_vocab, calc_bedroc, download_from_gcloud_bucket
from .load_data import load_rpob_data, load_rpoa_data, load_rpoc_data
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
import numpy as np
import ipdb
import json
import os
import wandb


# genes must be in order A, B, C
class Metrics:
    def __init__(self, embeddings_file, wt_seq_file, file_column_dict, beta=1, gene='rpoB', alpha=20,
                 results_fname="results", job_dir=False, wandb=False, model_type='attention', recalc_L1_diff=False):
        # ipdb.set_trace()
        self.vocab = generate_vocab(forbidden_aa='X')
        self.gene = gene
        self.recalc_L1_diff = recalc_L1_diff
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
        if os.path.exists(wt_seq_file):
            self.wt_seq_file = wt_seq_file
        else:
            if job_dir:
                self.wt_seq_file = download_from_gcloud_bucket(wt_seq_file)
        self.wt_seqs = read_fasta(self.wt_seq_file)
        self.model_type = model_type
        anchor_ids_dict = {
            'rpoB': 'tr|A0A0F6B9Y4|A0A0F6B9Y4_SALT1'
        }

        func_dict = {
            'rpoA': load_rpoa_data,
            'rpoB': load_rpob_data,
            'rpoC': load_rpoc_data
        }
        self.embedding_file = embeddings_file
        self.beta = beta
        if os.path.exists(self.embedding_file):
            self.weight_dict = torch.load(self.embedding_file, map_location=torch.device('cpu') if not torch.cuda.is_available()
                                          else torch.device("cuda: 0"))
        else:
            if job_dir:
                self.weight_dict = download_from_gcloud_bucket(self.embedding_file)
        ids = [seq.id for seq in self.wt_seqs]
        seqs = [str(seq.seq) for seq in self.wt_seqs]
        id_to_seq = {id: seq for id, seq in zip(ids, seqs)}
        self.gene_wt_dict = {gene: id_to_seq[anchor_ids_dict[gene]] for gene in list(anchor_ids_dict.keys())}

        with open(file_column_dict) as json_file:
            self.file_column_dictionary = json.load(json_file)
        # dictionary which shows the columns to compute metrics from files
        # format is {{filename: {mutation_column_name: name, eval_column_names: name}}}
        for anchor in list(anchor_ids_dict.keys()):
            self.df_dict = func_dict[anchor]()
        self.primary_fnames = [k for k, v in self.df_dict.items() if 'primary' in k]
        self.all_escape_muts = self.gather_all_escape_muts(self.primary_fnames, self.df_dict,
                                                           self.file_column_dictionary)
        self.comp_fnames = [k for k, v in self.df_dict.items() if 'comp' in k]
        self.alpha = 20
        self.results_fname = "{0}_{1}.json".format(results_fname, model_type)
        self.wandb = wandb

    def load_rpoa(self):
        raise NotImplementedError

    def embedding2grammar(self, gene_mut_seq_dict, seq, mut_name):
        embedding = torch.Tensor(gene_mut_seq_dict[seq]['embedding'])
        softmax_embedding = F.softmax(embedding, dim=1)
        pos = int(mut_name[1:-1])
        aa = mut_name[-1]
        if softmax_embedding.size()[0] == 1:
            softmax_embedding = softmax_embedding.squeeze(0)
        grammar = softmax_embedding[int(pos) - 1, list(self.vocab.keys()).index(aa) + 1]
        return grammar

    def fname2cscs(self, fname, df_dict, gene_mut_seq_dict, wt, baseline_diff=False, combinatoric=False, ori_wt=None,
                   primary_mut=None, recalc_l1_diff=False):
        mut_column = self.file_column_dictionary[fname]['mutation_column_name']
        mut_list = df_dict[fname]['df'][mut_column].tolist()
        # mut_list = [mut for mut in mut_list if mut not in self.removed_muts]
        semantics_list, grammar_list = [], []
        for i, mut in enumerate(mut_list):
            mut_seq = mut_abbrev_to_seq(mut, wt)
            if not combinatoric:
                grammar = self.embedding2grammar(gene_mut_seq_dict, mut_seq, mut)
                grammar_list.append(grammar)
                if not baseline_diff:
                    if recalc_l1_diff:
                        mut_embedding = torch.Tensor(gene_mut_seq_dict[mut_seq]['embedding'])
                        wt_embedding = torch.Tensor(gene_mut_seq_dict[wt]['embedding'])
                        semantics = torch.sqrt(
                            torch.sum(torch.square(wt_embedding - mut_embedding), dim=[0, 1])).tolist()
                        # semantics = abs(torch.sum(wt_embedding - mut_embedding, dim=[0, 1])).tolist()
                        # semantics = torch.sum(abs(wt_embedding - mut_embedding), dim=[0, 1]).tolist()
                        gene_mut_seq_dict[mut_seq]['l1_semantic_diff'] = semantics
                    else:
                        semantics = gene_mut_seq_dict[mut_seq]['l1_semantic_diff']
                else:
                    # semantics = abs(gene_mut_seq_dict[mut_seq]['l1_semantic_diff'] - \
                    #             gene_mut_seq_dict[wt]['l1_semantic_diff'])
                    mut_embedding = torch.Tensor(gene_mut_seq_dict[mut_seq]['embedding'])
                    wt_embedding = torch.Tensor(gene_mut_seq_dict[wt]['embedding'])
                    semantics = torch.sum(abs(wt_embedding - mut_embedding), dim=[0, 1]).tolist()
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
            for cscs_score, grammar_score, semantic_score, mutation in zip(cscs, grammar_list, semantics_list,
                                                                           mut_seqlist):
                gene_mut_seq_dict[mutation]['cscs_ranked'] = cscs_score
                gene_mut_seq_dict[mutation]['grammaticality'] = grammar_score
                gene_mut_seq_dict[mutation]['l1_semantic_diff'] = semantic_score
        return cscs, semantic_ranks, grammar_ranks, gene_mut_seq_dict

    def load_rpob(self):
        wt = self.gene_wt_dict['rpoB']
        rpob_mut_seq_dict = self.weight_dict[wt]

        for fname in self.primary_fnames:
            cscs, semantic, grammar, rpob_mut_seq_dict = self.fname2cscs(fname, self.df_dict, rpob_mut_seq_dict, wt,
                                                                         recalc_l1_diff=self.recalc_L1_diff)
            eval_columns = self.file_column_dictionary[fname]['eval_column_names']
            fname_cscs_results = {eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], cscs))
                                  for eval_column in eval_columns}
            fname_semantic_results = {eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], semantic))
                                      for eval_column in eval_columns}
            fname_grammar_results = {eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], grammar))
                                     for eval_column in eval_columns}

            self.results_dict['rpoB']['primary'][fname] = {}

            self.results_dict['rpoB']['primary'][fname]['n'] = self.df_dict[fname]['n']
            self.results_dict['rpoB']['primary'][fname]['cscs'] = fname_cscs_results
            self.results_dict['rpoB']['primary'][fname]['semantic'] = fname_semantic_results
            self.results_dict['rpoB']['primary'][fname]['grammar'] = fname_grammar_results

        for fname in self.comp_fnames:
            primary_mutation = self.df_dict[fname]['baseline_mut']
            primary_mut_seq = mut_abbrev_to_seq(primary_mutation, wt)
            single_cscs, single_semantic, single_grammar, rpob_mut_seq_dict = self.fname2cscs(fname, self.df_dict,
                                                                                              rpob_mut_seq_dict,
                                                                                              primary_mut_seq,
                                                                                              baseline_diff=True)
            comb_cscs, comb_semantic, comb_grammar, rpob_mut_seq_dict = self.fname2cscs(fname, self.df_dict,
                                                                                        rpob_mut_seq_dict,
                                                                                        primary_mut_seq,
                                                                                        combinatoric=True,
                                                                                        ori_wt=wt,
                                                                                        primary_mut=primary_mutation,
                                                                                        baseline_diff=True)
            eval_columns = self.file_column_dictionary[fname]['eval_column_names']
            single_cscs_results = {eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], single_cscs))
                                   for eval_column in eval_columns}
            single_semantic_results = {
                eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], single_semantic))
                for eval_column in eval_columns}
            single_grammar_results = {
                eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], single_grammar))
                for eval_column in eval_columns}

            self.results_dict['rpoB']['secondary']['single'][fname] = {}
            self.results_dict['rpoB']['secondary']['single'][fname]['cscs'] = single_cscs_results
            self.results_dict['rpoB']['secondary']['single'][fname]['semantic'] = single_semantic_results
            self.results_dict['rpoB']['secondary']['single'][fname]['grammar'] = single_grammar_results

            comb_cscs_results = {eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], comb_cscs))
                                 for eval_column in eval_columns}
            comb_semantic_results = {eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], comb_semantic))
                                     for eval_column in eval_columns}
            comb_grammar_results = {eval_column: list(spearmanr(self.df_dict[fname]['df'][eval_column], comb_grammar))
                                    for eval_column in eval_columns}

            self.results_dict['rpoB']['secondary']['combinatoric'][fname] = {}
            self.results_dict['rpoB']['secondary']['combinatoric'][fname]['cscs'] = comb_cscs_results
            self.results_dict['rpoB']['secondary']['combinatoric'][fname]['semantic'] = comb_semantic_results
            self.results_dict['rpoB']['secondary']['combinatoric'][fname]['grammar'] = comb_grammar_results
            self.results_dict['rpoB']['secondary']['combinatoric'][fname]['n'] = self.df_dict[fname]['n']

        self.weight_dict[wt] = rpob_mut_seq_dict
        torch.save(self.weight_dict, self.embedding_file)

        with open('results/{0}'.format(self.results_fname), 'w') as f:
            json.dump(self.results_dict, f, indent=4)
        return self.results_dict, rpob_mut_seq_dict

    @staticmethod
    def gather_all_escape_muts(primary_fnames, df_dict, file_column_dictionary):
        mutation_list = []
        for fname in primary_fnames:
            mut_column_name = file_column_dictionary[fname]['mutation_column_name']
            mutation_list.append(df_dict[fname]['df'][mut_column_name].tolist())
        mutations = list(itertools.chain(*mutation_list))
        mutations = list(set(mutations))
        # misc_mutations = ['I530II', 'DQ(516-517)DQDQ ', 'FM(514-515)FMFM', 'F514FF']
        # different_strain_mut = ["K504N", "V146F", "V146G", "V146W", "T563P"]
        # remove_mutations = misc_mutations + different_strain_mut
        # for mutation in remove_mutations:
        #     mutations.remove(mutation)
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
        list_primary_mutations = self.gather_all_escape_muts(self.primary_fnames, self.df_dict,
                                                             self.file_column_dictionary)

        grammar_list, semantic_list = [], []
        idx_list = []
        mut_seq_dict = {k: v for k, v in mut_seq_dict.items() if k != str(self.wt_seqs[0].seq)}
        for idx, (mut, _) in enumerate(mut_seq_dict.items()):
            if not mut == str(self.wt_seqs[0].seq):
                mut_name = mut_seq_dict[mut]['mut_abbrev']
                if mut_name in list_primary_mutations:
                    idx_list.append(idx)
                semantics = mut_seq_dict[mut]['l1_semantic_diff']
                grammar = self.embedding2grammar(mut_seq_dict, mut, mut_name)
                grammar_list.append(grammar.item())
                semantic_list.append(semantics)
            else:
                continue

        fig, ax = plt.subplots(2)
        ax[0].hist(grammar_list, bins=20)
        ax[0].set_title('grammaticality distribution')
        ax[1].hist(semantic_list, bins=20)
        ax[1].set_title('semantics distribution')
        fig.savefig('cscs_distributions.png')

        grammar_scaled = func_dict[grammatical_scaling]().fit_transform(np.array(grammar_list).reshape(-1, 1))
        semantics_scaled = func_dict[semantic_scaling]().fit_transform(np.array(semantic_list).reshape(-1, 1))

        cscs = grammar_scaled + self.beta * semantics_scaled
        cscs_scaled = np.array(func_dict[cscs_scaling]().fit_transform(cscs))

        cscs_scaled, semantics_scaled, grammar_scaled = np.squeeze(cscs_scaled, axis=1), \
                                                        np.squeeze(semantics_scaled, axis=1), \
                                                        np.squeeze(grammar_scaled, axis=1)

        y_true = np.zeros(len(grammar_list))
        y_true[idx_list] = 1

        cscs_aps = average_precision_score(y_true, cscs_scaled)
        semantic_aps = average_precision_score(y_true, semantics_scaled)
        grammar_aps = average_precision_score(y_true, grammar_scaled)

        list_arrays = [np.transpose(np.array([y_true, measure])) for measure in [cscs_scaled,
                                                                                 semantics_scaled, grammar_scaled]]

        sorted_arrays = [data_array[np.argsort(data_array[:, 1])] for data_array in list_arrays]

        cscs_bedroc, semantic_bedroc, grammar_bedroc = [calc_bedroc(array[:, 0], self.alpha) for array in sorted_arrays]

        self.results_dict[self.gene]['primary']['overall_escape_id'] = {
            'cscs BEDROC': cscs_bedroc,
            'semantic BEDROC': semantic_bedroc,
            'grammar BEDROC': grammar_bedroc,
            'cscs APS': cscs_aps,
            'semantic APS': semantic_aps,
            'grammar APS': grammar_aps
        }

        if self.wandb:
            wandb.log({
                '{0} cscs BEDROC'.format(self.model_type): cscs_bedroc,
                '{0} semantic BEDROC'.format(self.model_type): semantic_bedroc,
                '{0} grammar BEDROC'.format(self.model_type): grammar_bedroc,
                '{0} cscs APS'.format(self.model_type): cscs_aps,
                '{0} semantic APS'.format(self.model_type): semantic_aps,
                '{0} grammar APS'.format(self.model_type): grammar_aps
            })
        if not os.path.isdir('results'):
            os.mkdir('results')
        with open('results/{0}'.format(self.results_fname), 'w') as f:
            json.dump(self.results_dict, f, indent=4)
        if self.wandb:
            wandb.save('results/{0}'.format(self.results_fname))
        return cscs_aps, semantic_aps, grammar_aps


if __name__ == '__main__':
    ting = Metrics('saved_models/compressed_comb.pth', 'escape_validation/anchor_seqs.fasta',
                   'escape_validation/file_column_dict.json', results_fname='results')
    goo, too = ting.load_rpob()
    j, k, l = ting.escape_metrics(too, 'min_max', 'min_max', 'min_max')
    # hi = torch.load('saved_models/third_trial_47.pth',  map_location=torch.device('cpu'))

# script for new gene
# add embedding for wt
