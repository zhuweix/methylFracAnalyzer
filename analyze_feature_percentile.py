import os
import sys
import gc
import sqlite3
import tomllib
import logging

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyBigWig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(message)s")
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

PROMOTER_FLANK = 200


def load_sample_sheet(sample_fn: str, exp: str):
    sample_pd = pd.read_csv(sample_fn, sep='\t', names=['prefix', 'sample'])
    sample_list = sample_pd['sample'].values
    if exp == 'Live':
        data_points = [int(s[:-1]) for s in sample_list]
    elif exp == 'Nuclei':
        data_points = [np.round(float(s[:-1])/32,2) for s in sample_list]
    return sample_list, data_points


def search_bigwig(sample_list: str, bw_hg38_prefix: str, bw_t2t_prefix: str):
    hg38_dict = {}
    t2t_dict = {}
    for sample in sample_list:
        fn = f"{bw_hg38_prefix}.Filter_{sample}.bw"
        if not os.path.exists(fn):
            raise OSError('BigWig for %s: %s not found' %(sample, fn))
        hg38_dict[sample] = fn
        fn = f"{bw_t2t_prefix}.Filter_{sample}.bw"
        if not os.path.exists(fn):
            raise OSError('BigWig for %s: %s not found' %(sample, fn))
        t2t_dict[sample] = fn
    return hg38_dict, t2t_dict


def load_annotation(resource: dict, org: str):
    other_fn = resource['general']['other_feature_table']
    cen_elem_fn = resource['general']['cen_element_table']
    active_asat_fn = resource['general']['active_asat_table']
    cenpa_fn = resource['general']['cenpa_table']

    if org == 'MCF7':
        gene_fn = resource['MCF7']['gene_table']
        chromhmm_fn = resource['MCF7']['chromhmm_table']
    elif org == 'MCF10':
        gene_fn =  resource['MCF10']['gene_table']
        chromhmm_fn = resource['MCF10']['chromhmm_table']
    else:
        raise ValueError('Invalid Org: %s' %org)

    gene_pd = pd.read_csv(gene_fn, index_col=None)
    other_pd = pd.read_csv(other_fn, index_col=None, usecols=['Chrom', 'Type', 'Start', 'End', 'Strand'])
    cen_elem_pd = pd.read_csv(cen_elem_fn, index_col=None)
    active_asat_pd = pd.read_csv(active_asat_fn, index_col=None)
    cenpa_pd = pd.read_csv(cenpa_fn, index_col=None)
    chromhmm_pd = pd.read_csv(chromhmm_fn, index_col=None, usecols=['Chrom', 'Start', 'End', 'StateName'])

    # promoter and genebody

    promoter_pd = []
    for strand, tmp_pd in gene_pd.groupby('Strand'):
        if strand == '+':
            start_vals = tmp_pd['Start'].values - PROMOTER_FLANK
            end_vals = tmp_pd['Start'].values
            tmp_prom_pd = pd.DataFrame({'Chrom': tmp_pd['Chrom'],
                'Start': start_vals, 'End': end_vals, 'Strand': strand, })
            tmp_prom_pd['Is_ATAC_High'] = tmp_pd['Is_ATAC_High']
            tmp_prom_pd['TPM_Quantile'] = tmp_pd['TPM_Quantile']
            promoter_pd.append(tmp_prom_pd)
        else:
            start_vals = tmp_pd['End'].values 
            end_vals = tmp_pd['End'].values + PROMOTER_FLANK
            tmp_prom_pd = pd.DataFrame({'Chrom': tmp_pd['Chrom'],
                'Start': start_vals, 'End': end_vals, 'Strand': strand, })
            tmp_prom_pd['Is_ATAC_High'] = tmp_pd['Is_ATAC_High']
            tmp_prom_pd['TPM_Quantile'] = tmp_pd['TPM_Quantile']
            promoter_pd.append(tmp_prom_pd)           
    promoter_pd = pd.concat(promoter_pd)

    return promoter_pd, gene_pd, other_pd, cen_elem_pd, active_asat_pd, cenpa_pd, chromhmm_pd


def calculate_percentile(exp: str, dpni_dict: dict, sample_list: list, data_points: list):
    """
    Calculate percentiles of specific data points and store them in a dictionary.
    
    Parameters:
    - dpni_dict: dictionary of DpnI data
    - sample_list: list, the list of sample names
    
    Returns:
    - Dictionary containing percentiles ('q50', 'q25', 'q75', 'q15', 'q85', 'q5', 'q95') as keys 
      and corresponding percentile values as lists
    
    Note:
    This function calculates percentiles for each sample in the input DataFrame and stores them in a dictionary.
    """
    

    q50_list = []
    q25_list = []
    q75_list = []
    q15_list = []
    q85_list = []
    q5_list = []
    q95_list = []

    for sample in sample_list:
        dpni = dpni_dict[sample]
        if len(dpni) == 0:
            q50_list.append(np.nan)
            q25_list.append(np.nan)
            q75_list.append(np.nan)
            q15_list.append(np.nan)
            q85_list.append(np.nan)
            q5_list.append(np.nan)
            q95_list.append(np.nan)
            continue
        q50 = np.percentile(dpni, 50)
        q25 = np.percentile(dpni, 25)
        q75 = np.percentile(dpni, 75)
        q15 = np.percentile(dpni, 15)
        q85 = np.percentile(dpni, 85)
        q5 = np.percentile(dpni, 5)
        q95 = np.percentile(dpni, 95)
        q50_list.append(q50)
        q25_list.append(q25)
        q75_list.append(q75)
        q15_list.append(q15)
        q85_list.append(q85)
        q5_list.append(q5)
        q95_list.append(q95)

    if exp == 'Live':
        feat_data = pd.DataFrame({'Time (h)':data_points, 'q50': q50_list, 'q25': q25_list, 'q75': q75_list,'q15': q15_list,
                    'q85': q85_list, 'q05': q5_list, 'q95': q95_list,}) 
    else:
        feat_data = pd.DataFrame({'Dam (nM)':data_points, 'q50': q50_list, 'q25': q25_list, 'q75': q75_list,'q15': q15_list,
                    'q85': q85_list, 'q05': q5_list, 'q95': q95_list,})        

    return feat_data


def calculate_features(sample_list, bw_hg38_dict, bw_t2t_dict, 
    promoter_pd, genebody_pd, other_pd, cen_elem_pd, active_asat_pd, cenpa_pd, chromhmm_pd, org):
    chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    # Hg38: promoter, genebody and other features
    genome_dict = {}
    promoter_dict = {}
    gene_dict =  {}
    other_dict = {}
    transcript_level = ['NoTrans'] + [f'Q{i}' for i in range(1, 6)]
    other_feat_list = ['tRNA', 'enhancer', 'cpgIsland', 'silencer', 'origin_of_replication', 'centromere',]

    for sample in sample_list:
        bw_fn = bw_hg38_dict[sample]
        genome_data = []
        p_data_dict = {'active': [], 'inactive': []}
        for t in transcript_level:
            p_data_dict[t] = []
        g_data_dict = {'active': [], 'inactive': []}
        for t in transcript_level:
            g_data_dict[t] = []
        o_dict = {f: [] for f in other_feat_list}
        with pyBigWig.open(bw_fn) as bw:
            for chrom in chrom_list:
                size = bw.chroms()[chrom]
                tmp_dpni = bw.values(chrom, 0, size, numpy=True)

                tmp_prom_pd = promoter_pd.loc[promoter_pd['Chrom'] == chrom]
                tmp_gene_pd = genebody_pd.loc[genebody_pd['Chrom'] == chrom]
                tmp_other_pd = other_pd.loc[other_pd['Chrom'] == chrom]

                genome_data.append(tmp_dpni[~np.isnan(tmp_dpni)])
                # promoter
                for start, end, is_atac, tpm in zip(tmp_prom_pd['Start'], tmp_prom_pd['End'], tmp_prom_pd['Is_ATAC_High'], tmp_prom_pd['TPM_Quantile']):
                    tmp = tmp_dpni[start: end]
                    tmp = tmp[~np.isnan(tmp)]
                    if is_atac:
                        p_data_dict['active'].append(tmp)
                    else:
                        p_data_dict['inactive'].append(tmp)
                    p_data_dict[tpm].append(tmp)
                
                
                # genebody
                for start, end, is_atac, tpm in zip(tmp_gene_pd['Start'], tmp_gene_pd['End'], tmp_gene_pd['Is_ATAC_High'], tmp_gene_pd['TPM_Quantile']):
                    tmp = tmp_dpni[start: end]
                    tmp = tmp[~np.isnan(tmp)]
                    if is_atac:
                        g_data_dict['active'].append(tmp)
                    else:
                        g_data_dict['inactive'].append(tmp)
                    g_data_dict[tpm].append(tmp)
                # other feat
                for feat in other_feat_list:
                    tmp_feat_pd = tmp_other_pd.loc[tmp_other_pd['Type'] == feat]
                    for start, end in zip(tmp_feat_pd['Start'], tmp_feat_pd['End']):
                        tmp = tmp_dpni[start: end]
                        tmp = tmp[~np.isnan(tmp)]
                        o_dict[feat].append(tmp)
            genome_data = np.concatenate(genome_data)
            for k in p_data_dict:
                p_data_dict[k] = np.concatenate(p_data_dict[k])
            for k in g_data_dict:
                g_data_dict[k] = np.concatenate(g_data_dict[k])
            for feat in other_feat_list:
                o_dict[feat] = np.concatenate(o_dict[feat])

            genome_dict[sample] = genome_data
            promoter_dict[sample] = p_data_dict
            gene_dict[sample] = g_data_dict
            other_dict[sample] = o_dict
            logger.info('Calulation of features in Hg38 in %s is done.' %sample)
            gc.collect()

    # T2T: centromere
    cen_elem_dict = {}
    asat_dict = {}
    cepna_dict = {}
    chromhmm_dict = {}
    asat_chrom_dict = {}
    element_list = ['αSat-Active', 'αSat-Inactive', 'αSat-Other', 'HSat1',
        'HSat2', 'HSat3', 'βSat', 'rDNA', 'OtherSat', 'NonSat']
    sf_list = ['SF1', 'SF2', 'SF3', 'SF01']

    if org == 'MCF7':
        state_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2', 
                    'ZNF/Rpts', 'Het', 'Het2', 'ReprPc', 'Biv', 'NoMark']
    else:
        state_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2', 
                    'EnhWk', 'ZNF/Rpts', 'Het', 'ReprPc', 'Biv', 'NoMark']
    for sample in sample_list:
        bw_fn = bw_t2t_dict[sample]
        e_dict = {e: [] for e in element_list}
        a_dict = {s: [] for s in sf_list}
        ac_dict = {c: [] for c in chrom_list}
        c_dict = {s: [] for s in sf_list}
        c_dict['all'] = []
        ch_dict = {s: [] for s in state_list}
        with pyBigWig.open(bw_fn) as bw:
            for chrom in chrom_list:
                size = bw.chroms()[chrom]
                tmp_dpni = bw.values(chrom, 0, size, numpy=True)

                tmp_elem_pd = cen_elem_pd.loc[cen_elem_pd['Chrom'] == chrom]
                tmp_asat_pd = active_asat_pd.loc[active_asat_pd['Chrom'] == chrom]
                tmp_cenpa_pd = cenpa_pd.loc[cenpa_pd['Chrom'] == chrom]

                tmp_chromhmm_pd = chromhmm_pd.loc[chromhmm_pd['Chrom'] == chrom]

                for elem in element_list:
                    tmp_pd = tmp_elem_pd.loc[tmp_elem_pd['Class'] == elem]
                    for start, end in zip(tmp_pd['Start'], tmp_pd['End']):
                        tmp = tmp_dpni[start: end]
                        e_dict[elem].append(tmp[~np.isnan(tmp)])

                for s in sf_list:
                    tmp_pd = tmp_asat_pd.loc[tmp_asat_pd['HorClass'] == s]
                    for start, end in zip(tmp_pd['Start'], tmp_pd['End']):
                        tmp = tmp_dpni[start: end]
                        tmp = tmp[~np.isnan(tmp)]
                        a_dict[s].append(tmp)
                
                for start, end in zip(tmp_asat_pd['Start'], tmp_asat_pd['End']):
                    tmp = tmp_dpni[start: end]
                    tmp = tmp[~np.isnan(tmp)]
                    ac_dict[chrom].append(tmp)

                for s in sf_list:
                    tmp_pd = tmp_cenpa_pd.loc[tmp_cenpa_pd['HorClass'] == s]
                    for start, end in zip(tmp_pd['Start'], tmp_pd['End']):
                        tmp = tmp_dpni[start: end]
                        tmp = tmp[~np.isnan(tmp)]
                        c_dict[s].append(tmp)
                        c_dict['all'].append(tmp)

                for s in state_list:
                    tmp_pd = tmp_chromhmm_pd.loc[tmp_chromhmm_pd['StateName'] == s]
                    for start, end in zip(tmp_pd['Start'], tmp_pd['End']):
                        tmp = tmp_dpni[start: end]
                        tmp = tmp[~np.isnan(tmp)]
                        ch_dict[s].append(tmp)

    

            for k in e_dict.keys():
                e_dict[k] = np.concatenate(e_dict[k])
            for k in a_dict.keys():
                a_dict[k] = np.concatenate(a_dict[k])
            for k in c_dict.keys():
                c_dict[k] = np.concatenate(c_dict[k])
            for k in ch_dict.keys():
                ch_dict[k] = np.concatenate(ch_dict[k])
            for k in ac_dict.keys():
                ac_dict[k] = np.concatenate(ac_dict[k])
            cen_elem_dict[sample] = e_dict
            asat_dict[sample] = a_dict
            cepna_dict[sample] = c_dict
            chromhmm_dict[sample] = ch_dict
            asat_chrom_dict[sample] = ac_dict
            logger.info('Calulation of features in T2T in %s is done.' %sample)
            gc.collect()
    
    return genome_dict, promoter_dict, gene_dict, other_dict, cen_elem_dict, asat_dict, asat_chrom_dict, cepna_dict, chromhmm_dict
            

def plot_percentile_individual(plot_pd: pd.DataFrame,
                                datapoints: list,
                                figure_dirn: str, 
                                figure_name: str,
                                experiment: str):
    """
    Plots the percentiles of DpnI (%) for each type of annotated features 
    The source data of the figures are stored in the figure_source_dirn as well.

    Args:
        plot_pd (pd.DataFrame): The table of features and their corresponding percentiles.
        sample_list (list): The list of time after transduction.
        figure_dirn (str): The directory to save the figures.
        figure_source_dirn (str): The directory to save the source data of the figures.
        prefix (str): The prefix of the figure name.
        experiment (str): the type of samples Nuclei or Live

    Returns:
        None
    """
    
    figure_fn = os.path.join(figure_dirn, figure_name)
    q50_list = plot_pd['q50'].values
    q25_list = plot_pd['q25'].values
    q75_list = plot_pd['q75'].values
    q15_list = plot_pd['q15'].values
    q85_list = plot_pd['q85'].values
    q5_list = plot_pd['q05'].values
    q95_list = plot_pd['q95'].values
    xpos = datapoints

    g = sns.lineplot(x=xpos, y=q50_list, color='red', label='Median')
    plt.fill_between(xpos, q5_list, q95_list, alpha=0.09, color='red', label='5-95%')
    plt.fill_between(xpos, q15_list, q85_list, alpha=0.11, color='red', label='15-85%')
    plt.fill_between(xpos, q25_list, q75_list, alpha=0.15, color='red', label='25-75%')
    g.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    if experiment == 'Live':
        g.set_xlabel('Time after transduction (h)', fontsize='14')
    elif experiment == 'Nuclei':
        g.set_xlabel('Dam concentration (nM)', fontsize='14')
    g.set_ylabel('Fraction methylated (cut by DpnI) (%)', fontsize='12')

    g.set(xlim=(0, max(xpos)))
    g.set(ylim=(0, 100))
    g.set(yticks=np.arange(0, 101, 10), xticks=xpos)
    # set figure size
    g.figure.set_size_inches(6, 4)
    plt.savefig(figure_fn, dpi=300, facecolor='white', bbox_inches='tight', transparent=False)
    plt.close()


def plot_all_median(plot_pd: pd.DataFrame, feat_list: list, feat_name_list: list, datapoints: list,
                    figure_dirn: str, figure_source_dirn: str, figure_name: str, exp: str):
    figure_source_name = os.path.join(figure_source_dirn, figure_name.replace('.png', '.csv'))
    figure_name = os.path.join(figure_dirn, figure_name)
    xpos = datapoints
    if len(feat_list) <= 10:
        cmap = sns.color_palette('tab10')
    else:
        cmap = sns.color_palette('viridis', n_colors=len(feat_list))
    sns.set_palette(cmap)
    for feat, feat_name in zip(feat_list, feat_name_list):
        q50 = plot_pd.loc[plot_pd['Feature'] == feat]['q50'].values
        g = sns.lineplot(x=xpos, y=q50, label=feat_name)
    if len(feat_list) <= 10:
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Feature')
    else:
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Feature', ncol=2)
    if exp == 'Live':
        g.set_xlabel('Time after transduction (h)', fontsize='14')
    elif exp == 'Nuclei':
        g.set_xlabel('Dam concentration (nM)', fontsize='14')

    g.set_ylabel('Fraction methylated (cut by DpnI) (%)', fontsize='12')
    g.set(xlim=(0, max(xpos)))
    g.set(ylim=(0, 100))
    g.set(yticks=np.arange(0, 101, 10), xticks=xpos)
    # set figure size
    g.figure.set_size_inches(6, 4)
    plt.savefig(figure_name, dpi=300, facecolor='white', bbox_inches='tight', transparent=False)
    plt.close()
    plot_pd.to_csv(figure_source_name, index=False)



@click.command(help='Calculate and plot the percentile of Methylated Fraction in provided features')
@click.option('--configfile', '-c', type=str, default='', help='config file path')
@click.option('--resourcefile', '-R', type=str, default='config/resource.toml', help='resource config file path')
@click.option('--run', '-r', type=str, default='full', help='run mode: full, calc, plot for full run or calculation only or plotting only')
def main(configfile: str, resourcefile: str, run: str):
    if not os.path.exists(configfile):
        raise ValueError('Cannot find configfile %s' %configfile)
    with open(configfile, "rb") as f:
        config = tomllib.load(f)
    org = config['meta']['org']
    exp = config['meta']['exp']
    if exp not in ['Live', 'Nuclei']:
        raise ValueError('exp must be Live or Nuclei, found %s' %exp)
    sample_fn = config['sample']['sample_sheet']
    bw_hg38_prefix = config['bigwig']['bw_hg38_prefix']
    bw_t2t_prefix = config['bigwig']['bw_t2t_prefix']

    sample_list, data_points = load_sample_sheet(sample_fn, exp)

    figure_dir = config['output']['figure_dir']
    figure_source_dir = os.path.join(figure_dir, 'source')
    os.makedirs(figure_source_dir, exist_ok=True)

    figure_prefix = config['output']['figure_prefix']

    output_feat_fn = os.path.join(figure_source_dir, f"{figure_prefix}.hg38.feature.quantile.csv")
    output_cen_fn = os.path.join(figure_source_dir, f"{figure_prefix}.t2t.centromere.quantile.csv")
    output_chromhmm_fn = os.path.join(figure_source_dir, f"{figure_prefix}.t2t.chromhmm.quantile.csv")

    if run in ['full', 'calc']:
        # calculate the percentiles

        bw_hg38_dict, bw_t2t_dict = search_bigwig(sample_list, bw_hg38_prefix, bw_t2t_prefix)

        with open(resourcefile, "rb") as f:
            resource = tomllib.load(f)

        promoter_pd, genebody_pd, other_pd, cen_elem_pd, active_asat_pd, cenpa_pd, chromhmm_pd = load_annotation(resource, org)

        genome_dict, promoter_dict, gene_dict, other_dict, cen_elem_dict, asat_dict, asat_chrom_dict, cepna_dict, chromhmm_dict = calculate_features(
            sample_list, bw_hg38_dict, bw_t2t_dict, promoter_pd, genebody_pd, other_pd, cen_elem_pd, active_asat_pd, cenpa_pd, chromhmm_pd, org)

        feature_pd = []
        transcript_level = ['NoTrans'] + [f'Q{i}' for i in range(1, 6)]   
        # genome
        tmp_pd = calculate_percentile(exp, genome_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'Genome')
        feature_pd.append(tmp_pd)
        # promoter-all
        tmp_dict = {s: np.concatenate([promoter_dict[s]['inactive'],promoter_dict[s]['active']]) for s in sample_list}
        tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'Promoter')
        feature_pd.append(tmp_pd)
        # promoter-active
        tmp_dict = {s: promoter_dict[s]['active'] for s in sample_list}
        tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'Promoter Active')
        feature_pd.append(tmp_pd)
        # promoter-inactive
        tmp_dict = {s: promoter_dict[s]['inactive'] for s in sample_list}
        tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'Promoter Inactive')
        feature_pd.append(tmp_pd)
   

        for t in transcript_level:
            tmp_dict = {s: promoter_dict[s][t] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            tmp_pd.insert(0, 'Feature', 'Promoter '+t)
            feature_pd.append(tmp_pd)

        # genebody-all
        tmp_dict = {s: np.concatenate([gene_dict[s]['inactive'],gene_dict[s]['active']]) for s in sample_list}
        tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'Genebody')
        feature_pd.append(tmp_pd)        

        # genebody-active
        tmp_dict = {s: gene_dict[s]['active'] for s in sample_list}
        tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'Genebody Active')
        feature_pd.append(tmp_pd)
        # genebody-inactive
        tmp_dict = {s: gene_dict[s]['inactive'] for s in sample_list}
        tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'Genebody Inactive')
        feature_pd.append(tmp_pd)
        for t in transcript_level:
            tmp_dict = {s: gene_dict[s][t] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            tmp_pd.insert(0, 'Feature', 'Genebody '+t)
            feature_pd.append(tmp_pd)

        # other feature
        other_feat_list = ['tRNA', 'enhancer', 'cpgIsland', 'silencer', 'origin_of_replication', 'centromere',]
        for f in other_feat_list:
            tmp_dict = {s: other_dict[s][f] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            tmp_pd.insert(0, 'Feature', f)
            feature_pd.append(tmp_pd)    
        feature_pd = pd.concat(feature_pd)

        feature_pd.to_csv(output_feat_fn, index=False )

        centromoere_pd = []

        element_list = ['αSat-Active', 'αSat-Inactive', 'αSat-Other', 'HSat1', 
            'HSat2', 'HSat3', 'βSat', 'rDNA', 'OtherSat', 'NonSat']
        sf_list = ['SF1', 'SF2', 'SF3', 'SF01']

        for e in element_list:
            tmp_dict = {s: cen_elem_dict[s][e] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            e = e.replace('-', ' ')
            tmp_pd.insert(0, 'Feature', e)
            centromoere_pd.append(tmp_pd)
        
        for e in sf_list:
            tmp_dict = {s: asat_dict[s][e] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            tmp_pd.insert(0, 'Feature', 'αSat ' + e)
            centromoere_pd.append(tmp_pd)

        tmp_dict = {s: cepna_dict[s]['all'] for s in sample_list}
        tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
        tmp_pd.insert(0, 'Feature', 'CENPA')      
        centromoere_pd.append(tmp_pd)   

        for e in sf_list:
            tmp_dict = {s: cepna_dict[s][e] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            tmp_pd.insert(0, 'Feature', 'CENPA ' + e)     
            centromoere_pd.append(tmp_pd)            

        chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']
        # a-Sat in each chromosome
        for c in chrom_list:
            tmp_dict = {s: asat_chrom_dict[s][c] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            tmp_pd.insert(0, 'Feature', 'αSat ' + c)
            centromoere_pd.append(tmp_pd)

        centromoere_pd = pd.concat(centromoere_pd)
        centromoere_pd.to_csv(output_cen_fn, index=False)

        if org == 'MCF7':
            state_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2', 
                        'ZNF/Rpts', 'Het', 'Het2', 'ReprPc', 'Biv', 'NoMark']
        else:
            state_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2', 
                        'EnhWk', 'ZNF/Rpts', 'Het', 'ReprPc', 'Biv', 'NoMark']

        chromhmm_quantile_pd = []
        for e in state_list:
            tmp_dict = {s: chromhmm_dict[s][e] for s in sample_list}
            tmp_pd = calculate_percentile(exp, tmp_dict, sample_list, data_points)
            tmp_pd.insert(0, 'Feature', e)     
            chromhmm_quantile_pd.append(tmp_pd) 

        chromhmm_quantile_pd = pd.concat(chromhmm_quantile_pd)
        chromhmm_quantile_pd.to_csv(output_chromhmm_fn, index=False)

    # plot the figures
    if run in ['full', 'plot']:    
        if not os.path.exists(output_feat_fn):
            raise OSError('Cannot found %s, please run calc first.' % output_feat_fn)
        if not os.path.exists(output_cen_fn):
            raise OSError('Cannot found %s, please run calc first.' % output_cen_fn)
        if not os.path.exists(output_chromhmm_fn):
            raise OSError('Cannot found %s, please run calc first.' % output_chromhmm_fn)

        feat_pd = pd.read_csv(output_feat_fn, index_col=None)
        centromoere_pd = pd.read_csv(output_cen_fn, index_col=None)
        chromhmm_quantile_pd = pd.read_csv(output_chromhmm_fn, index_col=None)

        # individual figures
        figure_dir_ind = os.path.join(figure_dir, 'general')
        os.makedirs(figure_dir_ind, exist_ok=True)
        for feat, tmp_pd in feat_pd.groupby('Feature'):
            figname = f'{figure_prefix}.Feature.Quantile.{feat}.png'
            plot_percentile_individual(plot_pd=tmp_pd,
                                datapoints=data_points,
                                figure_dirn=figure_dir_ind, 
                                figure_name=figname,
                                experiment=exp)
        logger.info('General features are plotted')

        figure_dir_ind = os.path.join(figure_dir, 'centromere')
        os.makedirs(figure_dir_ind, exist_ok=True)
        for feat, tmp_pd in centromoere_pd.groupby('Feature'):
            figname = f'{figure_prefix}.Centromere.Quantile.{feat}.png'
            plot_percentile_individual(plot_pd=tmp_pd,
                                datapoints=data_points,
                                figure_dirn=figure_dir_ind, 
                                figure_name=figname,
                                experiment=exp)
        logger.info('Centromeric features are plotted')

        figure_dir_ind = os.path.join(figure_dir, 'chromhmm')
        os.makedirs(figure_dir_ind, exist_ok=True)
        for feat, tmp_pd in chromhmm_quantile_pd.groupby('Feature'):

            figname = f'{figure_prefix}.ChromHMM.Quantile.{feat}.png'
            figname = figname.replace('/', '_')

            plot_percentile_individual(plot_pd=tmp_pd,
                                datapoints=data_points,
                                figure_dirn=figure_dir_ind, 
                                figure_name=figname,
                                experiment=exp)
        logger.info('ChromHMM features are plotted')

        # median plot
        figure_dir_median = os.path.join(figure_dir, 'median')
        os.makedirs(figure_dir_median, exist_ok=True)

        feat_list = ['Promoter Active','Promoter Inactive', 'Genebody Active', 'Genebody Inactive']
        figure_name = f'{figure_prefix}.Gene.Median.byNDR.png'
        tmp_pd = feat_pd.loc[feat_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_list, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)

        feat_list = [f'Promoter Q{i}' for i in range(5, 0, -1)] + ['Promoter NoTrans']
        feat_names = [f'Q{i}' for i in range(5, 0, -1)] + ['NoTrans']
        figure_name = f'{figure_prefix}.Gene_Promoter.Median.byExpression.png'
        tmp_pd = feat_pd.loc[feat_pd['Feature'].isin(feat_list)]
        
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_names, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)
        feat_list = [f'Genebody Q{i}' for i in range(5, 0, -1)] + ['Genebody NoTrans']
        feat_names = [f'Q{i}' for i in range(5, 0, -1)] + ['NoTrans']
        figure_name = f'{figure_prefix}.Gene_Genebody.Median.byExpression.png'
        tmp_pd = feat_pd.loc[feat_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_names, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)

        feat_list = ["Promoter", "Genebody", 'tRNA', 'enhancer', 'cpgIsland', 'silencer', 'origin_of_replication', 'centromere',]
        feat_names = ["Promoter", "Genebody", 'tRNA', 'Enhancer', 'CpgIsland', 'Silencer', 'Origin of Replication', 'Centromere',]
        figure_name = f'{figure_prefix}.GeneralFeature.median.png'
        tmp_pd = feat_pd.loc[feat_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_names, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)


        if org == 'MCF7':
            active_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2',]            
            other_list = ['ZNF/Rpts', 'Het', 'Het2', 'ReprPc', 'Biv', 'NoMark']
        else:
            active_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2', 'EnhWk']           
            other_list = ['ZNF/Rpts', 'Het', 'ReprPc', 'Biv', 'NoMark']
        feat_list = active_list
        figure_name = f'{figure_prefix}.ChromHMM.Median.ActiveChromatin.png'
        tmp_pd = chromhmm_quantile_pd.loc[chromhmm_quantile_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_list, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)
        feat_list = other_list
        figure_name = f'{figure_prefix}.ChromHMM.Median.OtherChromatin.png'
        tmp_pd = chromhmm_quantile_pd.loc[chromhmm_quantile_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, figure_source_dirn=figure_source_dir, feat_name_list=feat_list, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_name=figure_name, exp=exp)

        feat_list = ['αSat Active', 'αSat Inactive', 'αSat Other', 'HSat1', 
            'HSat2', 'HSat3', 'βSat', 'rDNA', 'OtherSat', 'NonSat']
        figure_name = f'{figure_prefix}.Centromere.Median.CenElements.png'
        tmp_pd = centromoere_pd.loc[centromoere_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list,figure_source_dirn=figure_source_dir, feat_name_list=feat_list, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_name=figure_name, exp=exp)

        sf_list = ['SF1', 'SF2', 'SF3', 'SF01']
        feat_list = ['αSat Active'] + ['αSat ' + s for s in sf_list]
        figure_name = f'{figure_prefix}.Centromere.Median.Active_αSat.png'
        tmp_pd = centromoere_pd.loc[centromoere_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_list, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)        
        
        chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']
        feat_list = ['αSat ' + s for s in chrom_list]
        figure_name = f'{figure_prefix}.Centromere.Median.αSat.Chrom.png'
        tmp_pd = centromoere_pd.loc[centromoere_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_list, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)

        feat_list = ['CENPA'] + ['CENPA ' + s for s in sf_list]
        figure_name = f'{figure_prefix}.Centromere.Median.CENPA.png'
        tmp_pd = centromoere_pd.loc[centromoere_pd['Feature'].isin(feat_list)]
        plot_all_median(plot_pd=tmp_pd, feat_list=feat_list, feat_name_list=feat_list, datapoints=data_points,
                    figure_dirn=figure_dir_median, figure_source_dirn=figure_source_dir, figure_name=figure_name, exp=exp)     
        logger.info('Median plots are plotted') 


# if __name__ == '__main__':
#     main()