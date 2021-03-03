import pandas as pd
import ipdb
from .utils import download_from_gcloud_bucket


def load_rpob_data(job_dir=False):
    misc_mutations = ['I530II', 'DQ(516-517)DQDQ ', 'FM(514-515)FMFM', 'F514FF']
    different_strain_mut = ["K504N", "V146F", "V146G", "V146W", "T563P"]
    remove_mutations = misc_mutations + different_strain_mut

    list_filenames = [
        'rifr_comp_brandis.csv',
        'rifr_comp_brandis_hughes.csv',
        'rifr_primary_brandis.csv',
        'rifr_primary_brandis_pietsch.csv',
        'rifr_primary_brandis_pietsch_2.csv',
        'rifr_dms_replicate_a.txt',
        'rifr_dms_replicate_b.txt',
        'rifr_dms_replicate_c.txt'
    ]
    # rifr_primary_mutations = ['R529C', 'S531L']
    # comp_mutations = {} #compensatory secondary mutations (HCMs)
    # comp_mutations['R529C']
    df_dict = {}
    for fname in list_filenames:
        df_dict[fname] = {}

    #rifr_comp_brandis
    fname_1 = "escape_validation/rifr_comp_brandis.csv" if not job_dir else download_from_gcloud_bucket("rifr_comp_brandis.csv")
    df_rifr_comp_brandis = pd.read_csv(fname_1)
    df_rifr_comp_brandis = df_rifr_comp_brandis[df_rifr_comp_brandis['gene_segment']=='rpoB']
    df_dict['rifr_comp_brandis.csv']['df'] = df_rifr_comp_brandis[~df_rifr_comp_brandis["mutation"].isin(remove_mutations)]
    df_dict['rifr_comp_brandis.csv']['baseline_mut'] = 'R529C'
    df_dict['rifr_comp_brandis.csv']['n'] = len(df_rifr_comp_brandis)


    #rifr_comp_brandis_hughes
    fname_2 = 'escape_validation/rifr_comp_brandis_hughes.csv' if not job_dir else download_from_gcloud_bucket("rifr_comp_brandis_hughes.csv")
    df_rcbh = pd.read_csv(fname_2)
    df_rcbh = df_rcbh.loc[4:9]
    df_dict['rifr_comp_brandis_hughes.csv']['df'] = df_rcbh[~df_rcbh["comp_mutation"].isin(remove_mutations)]
    df_dict['rifr_comp_brandis_hughes.csv']['baseline_mut'] = 'S531L'
    df_dict['rifr_comp_brandis_hughes.csv']['n'] = len(df_rcbh)

    #rifr_primary_brandis
    fname_3 = 'escape_validation/rifr_primary_brandis.csv' if not job_dir else download_from_gcloud_bucket("rifr_primary_brandis.csv")
    df_pb = pd.read_csv(fname_3)
    df_pb = df_pb.loc[1:]
    df_dict['rifr_primary_brandis.csv']['df'] = df_pb[~df_pb["rpoB_mutation"].isin(remove_mutations)]
    df_dict['rifr_primary_brandis.csv']['n'] = len(df_pb)

    #rifr_primary_brandis_pietsch
    fname_4 = "escape_validation/rifr_primary_brandis_pietsch.csv" if not job_dir else download_from_gcloud_bucket("rifr_primary_brandis_pietsch.csv")
    df_pbp = pd.read_csv(fname_4)
    df_dict['rifr_primary_brandis_pietsch.csv']['df'] = df_pbp[~df_pbp['rpoB_mutation'].isin(remove_mutations)]
    df_dict['rifr_primary_brandis_pietsch.csv']['n'] = len(df_pbp)

    #rifr_primary_brandis_pietsch_2
    fname_5 = "escape_validation/rifr_primary_brandis_pietsch_2.csv" if not job_dir else download_from_gcloud_bucket("rifr_primary_brandis_pietsch_2.csv")
    df_pbp2 = pd.read_csv(fname_5)
    df_pbp2['is_indel'] = df_pbp2['rpoB_mutation'].apply(lambda x: '∆' in x)
    df_pbp2 = df_pbp2[df_pbp2['is_indel'] == False].loc[1:]
    df_dict['rifr_primary_brandis_pietsch_2.csv']['df'] = df_pbp2[~df_pbp2['rpoB_mutation'].isin(remove_mutations)]

    df_dict['rifr_primary_brandis_pietsch_2.csv']['n'] = len(df_pbp2)

    #rifr_dms_replicate_a
    fname_6 = "escape_validation/rifr_dms_replicate_a.txt" if not job_dir else download_from_gcloud_bucket("replicate_a.txt")
    df_dmsa = pd.read_csv(fname_6)
    df_dmsa = df_dmsa[df_dmsa['No_Non_Syn'] == 1]
    df_dict['rifr_dms_replicate_a.txt']['df'] = df_dmsa
    df_dict['rifr_dms_replicate_a.txt']['n'] = len(df_dmsa)

    # rifr_dms_replicate_b
    fname_7 = "escape_validation/rifr_dms_replicate_b.txt" if not job_dir else download_from_gcloud_bucket("replicate_b.txt")
    df_dmsb = pd.read_csv(fname_7)
    df_dmsb = df_dmsb[df_dmsb['No_Non_Syn'] == 1]
    df_dict['rifr_dms_replicate_b.txt']['df'] = df_dmsb
    df_dict['rifr_dms_replicate_b.txt']['n'] = len(df_dmsb)

    # rifr_dms_replicate_c
    fname_8 = "escape_validation/rifr_dms_replicate_c.txt" if not job_dir else download_from_gcloud_bucket("replicate_c.txt")
    df_dmsc = pd.read_csv(fname_8)
    df_dmsc = df_dmsc[df_dmsc['No_Non_Syn'] == 1]
    df_dict['rifr_dms_replicate_c.txt']['df'] = df_dmsc
    df_dict['rifr_dms_replicate_c.txt']['n'] = len(df_dmsc)

    return df_dict


def load_rpoa_data(job_dir=False):
    pass


def load_rpoc_data(job_dir=False):
    pass


def load_beta_lactamase():
    pass

