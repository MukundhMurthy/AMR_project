import pandas as pd
import ipdb

def load_rpob_data():
    list_filenames = [
        'rifr_comp_brandis.csv',
        'rifr_comp_brandis_hughes.csv',
        'rifr_primary_brandis.csv',
        'rifr_primary_brandis_pietsch.csv',
        'rifr_primary_brandis_pietsch_2.csv'
    ]
    # rifr_primary_mutations = ['R529C', 'S531L']
    # comp_mutations = {} #compensatory secondary mutations (HCMs)
    # comp_mutations['R529C']
    df_dict = {}
    for fname in list_filenames:
        df_dict[fname] = {}
    ipdb.set_trace()
    #rifr_comp_brandis
    df_rifr_comp_brandis = pd.read_csv('../escape_validation/rifr_comp_brandis.csv')
    df_rifr_comp_brandis = df_rifr_comp_brandis[df_rifr_comp_brandis['gene_segment']=='rpoB']
    df_dict['rifr_comp_brandis.csv']['df'] = df_rifr_comp_brandis
    df_dict['rifr_comp_brandis.csv']['baseline_mut'] = 'R529C'
    df_dict['rifr_comp_brandis.csv']['n'] = len(df_rifr_comp_brandis)


    #rifr_comp_brandis_hughes
    df_rcbh = pd.read_csv('../escape_validation/rifr_comp_brandis_hughes.csv')
    df_rcbh = df_rcbh.loc[4:9]
    df_dict['rifr_comp_brandis_hughes.csv']['df'] = df_rcbh
    df_dict['rifr_comp_brandis_hughes.csv']['baseline_mut'] = 'S531L'
    df_dict['rifr_comp_brandis_hughes.csv']['n'] = len(df_rcbh)

    #rifr_primary_brandis
    df_pb = pd.read_csv('../escape_validation/rifr_primary_brandis.csv')
    df_pb = df_pb.loc[1:]
    df_dict['rifr_primary_brandis.csv']['df'] = df_pb
    df_dict['rifr_primary_brandis.csv']['n'] = len(df_pb)

    #rifr_primary_brandis_pietsch
    df_pbp = pd.read_csv('../escape_validation/rifr_primary_brandis_pietsch.csv')
    df_dict['rifr_primary_brandis_pietsch.csv']['df'] = df_pbp
    df_dict['rifr_primary_brandis_pietsch.csv']['n'] = len(df_pbp)

    #rifr_primary_brandis_pietsch_2
    df_pbp2 = pd.read_csv('../escape_validation/rifr_primary_brandis_pietsch_2.csv')
    df_pbp2['is_indel'] = df_pbp2['rpoB_mutation'].apply(lambda x: '∆' in x)
    df_pbp2 = df_pbp2[df_pbp2['is_indel'] == False]
    df_dict['rifr_primary_brandis_pietsch_2.csv']['df'] = df_pbp2
    df_dict['rifr_primary_brandis_pietsch.csv']['n'] = len(df_pbp2)

    return df_dict


def load_beta_lactamase():
    pass

if __name__ == '__main__':
    load_rpob()