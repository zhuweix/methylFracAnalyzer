import os
import sys
import gc
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

FLANK = 1000
WINDOW = 21
MNASE_REL = 30

def load_sample_sheet(sample_fn: str, exp: str):
    sample_pd = pd.read_csv(sample_fn, sep='\t', names=['prefix', 'sample'])
    sample_list = sample_pd['sample'].values
    if exp == 'Live':
        data_points = [int(s[:-1]) for s in sample_list]
    elif exp == 'Nuclei':
        data_points = [np.round(float(s[:-1])/32,2) for s in sample_list]
    return sample_list, data_points


def search_bigwig(sample_list: str, bw_hg38_prefix: str):
    hg38_dict = {}
    for sample in sample_list:
        fn = f"{bw_hg38_prefix}.Filter_{sample}.bw"
        if not os.path.exists(fn):
            raise OSError('BigWig for %s: %s not found' %(sample, fn))
        hg38_dict[sample] = fn
    return hg38_dict

def load_annotation(resource: dict, org: str):
    if org == 'MCF7':
        gene_fn = resource['MCF7']['gene_table']
        ctcf_fn = resource['MCF7']['ctcf_table']
        mnase_gene_fn = resource['MCF7']['mnase_smooth_gene_table']
        mnase_ctcf_fn = resource['MCF7']['mnase_smooth_ctcf_table']
    elif org == 'MCF10':
        gene_fn =  resource['MCF10']['gene_table']
        ctcf_fn = resource['MCF10']['ctcf_table']
        mnase_gene_fn = resource['MCF10']['mnase_smooth_gene_table']
        mnase_ctcf_fn = resource['MCF10']['mnase_smooth_ctcf_table']        
    else:
        raise ValueError('Invalid Org: %s' %org)

    gene_pd = pd.read_csv(gene_fn, index_col=None)
    ctcf_pd = pd.read_csv(ctcf_fn, index_col=None)
    mnase_gene_pd = pd.read_csv(mnase_gene_fn, index_col=None)
    mnase_ctcf_pd = pd.read_csv(mnase_ctcf_fn, index_col=None)
    return gene_pd, ctcf_pd, mnase_gene_pd, mnase_ctcf_pd

def calc_gene_phasing(sample_list, bw_hg38_dict, gene_pd, data_points, gene_active_fn, gene_inactive_fn, mnase_pd):
    chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']    
    conv = np.ones(WINDOW) / WINDOW
    xpos = np.arange(-FLANK, FLANK)
    gene_active_pd = pd.DataFrame({'pos': xpos})   
    gene_inactive_pd = pd.DataFrame({'pos': xpos})  
    for sample, da in zip(sample_list, data_points):
        bw_fn = bw_hg38_dict[sample]
        active_array = []
        inactive_array = []
        with pyBigWig.open(bw_fn) as bw:
            for chrom in chrom_list:
                size = bw.chroms()[chrom]
                tmp_dpni = bw.values(chrom, 0, size, numpy=True)
                tmp_gene_pd = gene_pd.loc[gene_pd['Chrom'] == chrom]
                for start, end, is_atac, strand in zip(tmp_gene_pd['Start'], 
                    tmp_gene_pd['End'], tmp_gene_pd['Is_ATAC_High'], tmp_gene_pd['Strand']):
                    if strand == '+':
                        left = start - FLANK -WINDOW //2
                        right = start + FLANK + WINDOW //2
                        tmp = tmp_dpni[left: right].copy()
                    else:
                        left = end - FLANK -WINDOW //2
                        right = end + FLANK + WINDOW //2
                        tmp = tmp_dpni[left: right].copy()
                        tmp = tmp[::-1]
                    if is_atac:
                        active_array.append(tmp)
                    else:
                        inactive_array.append(tmp)
        active_array = np.nanmean(np.vstack(active_array), axis=0)
        inactive_array = np.nanmean(np.vstack(inactive_array), axis=0)

        inval = np.isnan(active_array)
        active_array[inval] = np.interp(np.where(inval)[0], np.where(~inval)[0], active_array[~inval])
        smooth_active = np.convolve(active_array, conv, 'valid')

        inval = np.isnan(inactive_array)
        inactive_array[inval] = np.interp(np.where(inval)[0], np.where(~inval)[0], inactive_array[~inval])
        smooth_inactive = np.convolve(inactive_array, conv, 'valid')

        gene_active_pd[str(da)]= smooth_active  
        gene_inactive_pd[str(da)]= smooth_inactive
        logger.info('Gene phasing in Sample %s is calculated' %sample) 

    gene_active_pd["MNase"]= mnase_pd['Active'] * MNASE_REL
    gene_inactive_pd["MNase"]= mnase_pd['Inactive'] * MNASE_REL
   

    gene_active_pd.to_csv(gene_active_fn, index=False)
    gene_inactive_pd.to_csv(gene_inactive_fn, index=False)

def calc_ctcf_phasing(sample_list, bw_hg38_dict, ctcf_pd, data_points, ctcf_phase_fn, mnase_pd):
    chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']    
    conv = np.ones(WINDOW) / WINDOW
    xpos = np.arange(-FLANK, FLANK)
    ctcf_phasing_pd = pd.DataFrame({'pos': xpos})    
    for sample, da in zip(sample_list, data_points):
        bw_fn = bw_hg38_dict[sample]
        ctcf_array = []
        with pyBigWig.open(bw_fn) as bw:
            for chrom in chrom_list:
                size = bw.chroms()[chrom]
                tmp_dpni = bw.values(chrom, 0, size, numpy=True)
                tmp_ctcf_pd = ctcf_pd.loc[ctcf_pd['Chrom'] == chrom]
                for m_pos, strand in zip(tmp_ctcf_pd['Motif_Pos'], tmp_ctcf_pd['Strand']):
                    left = m_pos - FLANK -WINDOW //2
                    right = m_pos + FLANK + WINDOW //2
                    tmp = tmp_dpni[left: right].copy()                    
                    if strand == '-':
                        tmp = tmp[::-1]
                    ctcf_array.append(tmp)
        ctcf_array = np.nanmean(np.vstack(ctcf_array), axis=0)

        inval = np.isnan(ctcf_array)
        ctcf_array[inval] = np.interp(np.where(inval)[0], np.where(~inval)[0], ctcf_array[~inval])
        smooth_ctcf = np.convolve(ctcf_array, conv, 'valid')
        ctcf_phasing_pd[str(da)]= smooth_ctcf
        logger.info('CTCF phasing in Sample %s is calculated' %sample)
    ctcf_phasing_pd['MNase'] = mnase_pd['MNase'] * MNASE_REL
    ctcf_phasing_pd.to_csv(ctcf_phase_fn, index=False)


def plot_gene_phasing(gene_active_fn, gene_inactive_fn, gene_active_fig, gene_inactive_fig, data_points, exp):
    # active gene
    data_pd = pd.read_csv(gene_active_fn, index_col=False)
    xpos = data_pd['pos']
    for da in data_points:
        da = str(da)
        value = data_pd[da]
        g = sns.lineplot(x=xpos, y=value, label=da, lw=.5)
    value = data_pd['MNase']        
    plt.fill_between(xpos, np.zeros(len(xpos)), value, label='MNase', color='.6', alpha=.5)         
    if exp == 'Live':
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Time (h)')
    else:
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Dam (nM)')

             
    g.set_ylabel('Fraction methylated (cut by DpnI) (%)', fontsize='12')
    g.set_xlabel('Relative to TSS (bp)', fontsize='14')
    g.set(xlim=(-FLANK, FLANK), xticks=np.arange(-FLANK, FLANK+1, 200))
    g.set(ylim=(0, 100))
    g.set(yticks=np.arange(0, 101, 10))
    # set figure size
    g.figure.set_size_inches(5.2, 4)
    plt.savefig(gene_active_fig, dpi=300, facecolor='white', bbox_inches='tight', transparent=False)
    plt.close()

    # inactive gene
    data_pd = pd.read_csv(gene_inactive_fn, index_col=False)
    xpos = data_pd['pos']
    for da in data_points:
        da = str(da)
        value = data_pd[da]
        g = sns.lineplot(x=xpos, y=value, label=da, lw=.5)
    value = data_pd['MNase']
    plt.fill_between(xpos, np.zeros(len(xpos)), value, label='MNase', color='.6', alpha=.5)        
    if exp == 'Live':
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Time (h)')
    else:
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Dam (nM)')        
    g.set_ylabel('Fraction methylated (cut by DpnI) (%)', fontsize='12')
    g.set_xlabel('Relative to TSS (bp)', fontsize='14')
    g.set(xlim=(-FLANK, FLANK), xticks=np.arange(-FLANK, FLANK+1, 200))
    g.set(ylim=(0, 100))
    g.set(yticks=np.arange(0, 101, 10))
    # set figure size
    g.figure.set_size_inches(5.2, 4)
    plt.savefig(gene_inactive_fig, dpi=300, facecolor='white', bbox_inches='tight', transparent=False)
    plt.close()

def plot_ctcf_phasing(ctcf_fn, gene_ctcf_fig, data_points, exp):
    # active gene
    data_pd = pd.read_csv(ctcf_fn, index_col=False)
    xpos = data_pd['pos']
    for da in data_points:
        da = str(da)
        value = data_pd[da]
        g = sns.lineplot(x=xpos, y=value, label=da, lw=.5)
    value = data_pd['MNase']
    plt.fill_between(xpos, np.zeros(len(xpos)), value, label='MNase', color='.6', alpha=.5)               
    if exp == 'Live':
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Time (h)')
    else:
        g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Dam (nM)')           
    g.set_ylabel('Fraction methylated (cut by DpnI) (%)', fontsize='12')
    g.set_xlabel('Relative to Motif (bp)', fontsize='14')
    g.set(xlim=(-FLANK, FLANK), xticks=np.arange(-FLANK, FLANK+1, 200))
    g.set(ylim=(0, 100))
    g.set(yticks=np.arange(0, 101, 10))
    # set figure size
    g.figure.set_size_inches(5.2, 4)
    plt.savefig(gene_ctcf_fig, dpi=300, facecolor='white', bbox_inches='tight', transparent=False)
    plt.close()


@click.command(help='Calculate and plot the phasing of methylated fraction relative to TSS in genes and relative to CTCF motifs')
@click.option('--configfile', '-c', type=str, default='', help='config file path')
@click.option('--resourcefile', '-R', type=str, default='config/resource.toml', help='resource config file path')
@click.option('--run', '-r', type=str, default='full', help='run mode: full, calc, plot for full run or calculation only or plotting only')
def main(configfile: str, resourcefile: str, run: str):
    if not configfile:
        click.echo(main.get_help(click.get_current_context()))
        return 
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

    sample_list, data_points = load_sample_sheet(sample_fn, exp)

    figure_dir = config['output']['figure_dir']
    figure_source_dir = os.path.join(figure_dir, 'source')
    os.makedirs(figure_source_dir, exist_ok=True)
    figure_dir = os.path.join(figure_dir, 'phasing')
    os.makedirs(figure_dir, exist_ok=True)    
    figure_prefix = config['output']['figure_prefix']

    gene_active_fn = os.path.join(figure_source_dir, f'{figure_prefix}.Gene.Phasing.Active.1kb.smooth_21bp.csv')
    gene_inactive_fn = os.path.join(figure_source_dir, f'{figure_prefix}.Gene.Phasing.Inactive.1kb.smooth_21bp.csv') 
    ctcf_phasing_fn = os.path.join(figure_source_dir, f'{figure_prefix}.CTCF.Phasing.1kb.smooth_21bp.csv') 
    
    if run in ['full', 'calc']:
        bw_hg38_dict = search_bigwig(sample_list, bw_hg38_prefix)

        with open(resourcefile, "rb") as f:
                resource = tomllib.load(f)

        gene_pd, ctcf_pd, mnase_gene_pd, mnase_ctcf_pd = load_annotation(resource, org)

        calc_gene_phasing(sample_list, bw_hg38_dict, gene_pd, data_points, gene_active_fn, gene_inactive_fn, mnase_gene_pd)
        logger.info('Calculation of Gene Phasing is done.')

        calc_ctcf_phasing(sample_list, bw_hg38_dict, ctcf_pd, data_points, ctcf_phasing_fn,  mnase_ctcf_pd)
        logger.info('Calculation of CTCF Phasing is done.')
    
    if run in ['full', 'plot']:
        gene_active_fn = os.path.join(figure_source_dir, f'{figure_prefix}.Gene.Phasing.Active.1kb.smooth_21bp.csv')
        gene_inactive_fn = os.path.join(figure_source_dir, f'{figure_prefix}.Gene.Phasing.Inactive.1kb.smooth_21bp.csv') 
        ctcf_phasing_fn = os.path.join(figure_source_dir, f'{figure_prefix}.CTCF.Phasing.1kb.smooth_21bp.csv') 
        if not os.path.isfile(gene_active_fn):
            raise OSError('Cannot found %s, please run calc first.' % gene_active_fn)
        if not os.path.isfile(gene_inactive_fn):
            raise OSError('Cannot found %s, please run calc first.' % gene_inactive_fn)
        if not os.path.isfile(ctcf_phasing_fn):
            raise OSError('Cannot found %s, please run calc first.' % ctcf_phasing_fn)

        gene_active_figure = os.path.join(figure_dir, f'{figure_prefix}.Gene.Phasing.Active.1kb.smooth_21bp.png')
        gene_inactive_figure = os.path.join(figure_dir, f'{figure_prefix}.Gene.Phasing.Inactive.1kb.smooth_21bp.png')
        ctcf_figure = os.path.join(figure_dir, f'{figure_prefix}.CTCF.Phasing.1kb.smooth_21bp.png')

        plot_gene_phasing(gene_active_fn, gene_inactive_fn, gene_active_figure, gene_inactive_figure, 
                        data_points, exp)
        plot_ctcf_phasing(ctcf_phasing_fn, ctcf_figure, data_points, exp)
        logger.info('Phasing plots are done')


if __name__ == '__main__':
    main()  
