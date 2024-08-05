import os
import sys
import gc
import tomllib
import logging
import pickle

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pyBigWig
import statsmodels.api as sm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(message)s")
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

WINDOW = 100_000

def load_sample_sheet(sample_fn: str, exp: str):
    sample_pd = pd.read_csv(sample_fn, sep='\t', names=['prefix', 'sample'])
    sample_list = sample_pd['sample'].values
    if exp == 'Live':
        data_points = [int(s[:-1]) for s in sample_list]
    elif exp == 'Nuclei':
        data_points = [np.round(float(s[:-1])/32,2) for s in sample_list]
    return sample_list, data_points


def search_bigwig(sample_list: str, bw_t2t_prefix: str):
    t2t_dict = {}
    for sample in sample_list:
        fn = f"{bw_t2t_prefix}.Filter_{sample}.bw"
        if not os.path.exists(fn):
            raise OSError('BigWig for %s: %s not found' %(sample, fn))
        t2t_dict[sample] = fn
    return t2t_dict

def calc_genome_map_frac(bw_dict, sample_list, data_points,
        figure_source_fn, exp):
    if exp == 'Live':
        sample_header = 'Time (h)'
    else:
        sample_header = 'Dam (nM)'
    chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    genome_map_frac = []
    for sample, da in zip(sample_list, data_points):
        if da < 1e-6:
            continue
        bw_fn = bw_dict[sample]
        with pyBigWig.open(bw_fn) as bw:
            for chrom in chrom_list:
                size = bw.chroms()[chrom]
                tmp_dpni = bw.values(chrom, 0, size, numpy=True)
                tmp_dpni = tmp_dpni[:size //WINDOW *WINDOW]
                tmp_dpni = np.nanmean(tmp_dpni.reshape(-1, WINDOW), axis=1)   
                genome_map_frac.append(pd.DataFrame({
                    sample_header: da,
                    'Chrom': chrom,
                    'BinPos': np.arange(0, len(tmp_dpni)) * WINDOW,
                    'Frac': tmp_dpni
                }))
        logger.info('Calucation of %s is done' %sample)
    genome_map_frac = pd.concat(genome_map_frac)
    genome_map_frac.to_csv(figure_source_fn, index=False)


def plot_genome_map(genome_frac_pd, cen_pos_pd, figure_prefix, data_points, exp):
    chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    if exp == 'Live':
        sample_header = 'Time (h)'
        figure_prefix = figure_prefix + '.time_'
    else:
        sample_header = 'Dam (nM)'
        figure_prefix = figure_prefix + '.dam_'
    max_size = genome_frac_pd['BinPos'].max() // WINDOW
    for da in data_points:
        if da < 1e-6:
            continue
        tmp_frac_pd = genome_frac_pd[genome_frac_pd[sample_header] == da]
        tmp_array = np.empty((23, max_size+1))
        tmp_array[:] = np.nan

        for i, chrom in enumerate(chrom_list):
            tmp_pd = tmp_frac_pd[tmp_frac_pd['Chrom'] == chrom]
            pos = tmp_pd['BinPos'].values // WINDOW
            frac = tmp_pd['Frac'].values
            tmp_array[i, pos] = frac
            tmp_array[i, np.max(pos):] = -1

    # set default font size to 14
        plt.rcParams.update({'font.size': 12})
        cmap = sns.color_palette('viridis', as_cmap=True)
        cmap.set_under('w')
        if exp == 'Live':
            g = sns.heatmap(data=tmp_array, vmin=0, vmax=100,cmap=cmap,  cbar_kws={'label':'Fraction methylated (Cut by DpnI)%', }, 
                            yticklabels=chrom_list,)
        else:
            g = sns.heatmap(data=tmp_array, vmin=0, vmax=60,cmap=cmap,  cbar_kws={'label':'Fraction methylated (Cut by DpnI)%', }, 
                            yticklabels=chrom_list,)

        # set title font size of cbar
        g.figure.axes[-1].yaxis.label.set_size(14)

        # highlight centromere
        for i, chrom in enumerate(chrom_list):
            tmp_pos = cen_pos_pd[cen_pos_pd['Chrom'] == chrom].copy()
            tmp_pos['Start'] = tmp_pos['Start'] // WINDOW
            tmp_pos['End'] = tmp_pos['End'] // WINDOW + 1
            for s, e in zip(tmp_pos['Start'], tmp_pos['End']):
                pat = mpatches.Rectangle((s, i), e-s, 1, linewidth=1.5, edgecolor='red', facecolor='none')
                g.figure.axes[0].add_patch(pat)

        # set title font size of cbar
        g.figure.axes[-1].yaxis.label.set_size(14)
        g.figure.axes[0].xaxis.label.set_size(14)
        g.set(facecolor='.7')
        _ = g.set(xticks=np.arange(0, max_size+1, 20_000_000 // WINDOW),
                xticklabels=np.arange(0, max_size * WINDOW // 1000_000 + 1, 20),
                xlabel='Chromosome Position (Mb)')
        plt.gcf().set_size_inches(10, 8)
        plt.savefig(f"{figure_prefix}{da}.png", dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        plt.close()
        logger.info('Plot of %s is done' %da)


@click.command(help='Calculate and plot the methylated fraction in 100 kb bins of the genome')
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
    with open(resourcefile, 'rb') as f:
        resource = tomllib.load(f)

    org = config['meta']['org']
    exp = config['meta']['exp']

    sample_fn = config['sample']['sample_sheet']
    bw_t2t_prefix = config['bigwig']['bw_t2t_prefix']
    sample_list, data_points = load_sample_sheet(sample_fn, exp)
    cen_loc_fn = resource['general']['cen_loc_table']

    figure_dir = config['output']['figure_dir']
    figure_source_dir = os.path.join(figure_dir, 'source')
    os.makedirs(figure_source_dir, exist_ok=True)

    figure_prefix = config['output']['figure_prefix']
    figure_dir = os.path.join(figure_dir, 'frac_map')
    os.makedirs(figure_dir, exist_ok=True)    


    figure_source_fn = os.path.join(figure_source_dir, f'{figure_prefix}.GenomeFracMap.100kb.csv')
    figure_prefix = os.path.join(figure_dir, f'{figure_prefix}.GenomeFracMap.100kb')

    if run in ['full', 'calc']:
        bw_t2t_dict = search_bigwig(sample_list, bw_t2t_prefix)
        calc_genome_map_frac(bw_t2t_dict, sample_list, data_points,
            figure_source_fn, exp)

    if run in ['full', 'plot']:
        if not os.path.isfile(figure_source_fn):
            raise OSError('Cannot find %s, please calculate rate again' %figure_source_fn)
        genome_frac_pd = pd.read_csv(figure_source_fn, index_col=False)
        cen_pos_pd = pd.read_csv(cen_loc_fn, index_col=False)
        plot_genome_map(genome_frac_pd, cen_pos_pd, figure_prefix, data_points, exp)

if __name__ == '__main__':
    main()