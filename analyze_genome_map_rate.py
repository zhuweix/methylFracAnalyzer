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

def calc_kinetic_rate(data_row, time_points):
    """
    Calculate the kinetic rate constant using linear regression.

    Parameters:
    - data_row: A 1D array containing data points.
    - time_points: A 1D array containing time points.

    Returns:
    - The calculated kinetic rate constant.
    """
    valid_pos = ~np.isnan(data_row)
    time_points = time_points[valid_pos]
    if time_points.shape[0] < 2:
        return np.nan
    data_row = data_row[valid_pos]
    X = time_points
    y = data_row.reshape(-1, 1)
    if np.any(np.isnan(y)):
        return np.nan

    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    _, k = result.params
    return -k


def calc_genome_map_rate(bw_dict, sample_list, data_points, figure_source_fn, figure_source_data):
    chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    if not os.path.isfile(figure_source_data):
        genome_time_dict = {c: [] for c in chrom_list}
        time_list = []
        genome_medians = []
        for sample, da in zip(sample_list, data_points):
            if da < 1e-6:
                continue
            time_list.append(da)
            bw_fn = bw_dict[sample]
            tmp_med = []
            with pyBigWig.open(bw_fn) as bw:
                for chrom in chrom_list:
                    size = bw.chroms()[chrom]
                    tmp_dpni = bw.values(chrom, 0, size, numpy=True)
                    tmp_dpni = tmp_dpni[:size //WINDOW *WINDOW]
                    tmp_dpni = np.nanmean(tmp_dpni.reshape(-1, WINDOW), axis=1)
                    tmp_dpni = 1 - tmp_dpni/100
                    genome_time_dict[chrom].append(tmp_dpni)
                    tmp_med.append(tmp_dpni)
            tmp_med = np.concatenate(tmp_med)
            genome_medians.append(np.nanmedian(tmp_med))
            gc.collect()
        time_list = np.array(time_list)
        genome_medians = np.log(np.array(genome_medians))
        rate_genome = calc_kinetic_rate(genome_medians, time_list)
        genome_time_dict['GenomeRate'] = rate_genome
        with open(figure_source_data, 'wb') as filep:
            pickle.dump(genome_time_dict, filep)
    else:
        time_list = []
        genome_medians = []
        for da in data_points:
            if da < 1e-6:
                continue
            time_list.append(da)     
        time_list = np.array(time_list)   
        with open(figure_source_data, 'rb') as filep:
            genome_time_dict = pickle.load(filep)
        rate_genome = genome_time_dict['GenomeRate']

    logger.info('The Genome Methyaltion is loaded, Median Methyaltion Rate: %.2f' %rate_genome)


    genome_rate_pd = []
    for chrom in chrom_list:
        tmp_dpni = np.vstack(genome_time_dict[chrom]).T
        mask = tmp_dpni < 0.01
        tmp_dpni[mask] = 0.01
        tmp_dpni = np.log(tmp_dpni)
        num_high = np.sum(mask, axis=1)
        # mask for dots over .99 for saturation
        if np.sum(num_high > 1) > 0:
            high_row = np.where(num_high > 1)[0]
            for r in high_row:
                high_pos = np.where(mask[r, :])[0][0]
                tmp_dpni[r, high_pos+1:] = np.nan

        rates = np.apply_along_axis(calc_kinetic_rate, 1, tmp_dpni, time_list) / rate_genome
        genome_rate_pd.append(pd.DataFrame({'Chrom': chrom, 
            'BinPos': np.arange(0, len(rates) * WINDOW, WINDOW), 'RelativeRate': rates}))
    genome_rate_pd = pd.concat(genome_rate_pd, ignore_index=True)
    genome_rate_pd.to_csv(figure_source_fn, index=False)


def plot_genome_map(genome_rate_pd, cen_pos_pd, figure_fn):
    max_size = genome_rate_pd['BinPos'].max() // WINDOW
    tmp_array = np.empty((23, max_size+1))
    tmp_array[:] = np.nan
    chrom_list = ['chr'+str(i) for i in range(1, 23)] + ['chrX']
    for i, chrom in enumerate(chrom_list):
        tmp_pd = genome_rate_pd[genome_rate_pd['Chrom'] == chrom]
        pos = tmp_pd['BinPos'].values // WINDOW
        rate = tmp_pd['RelativeRate'].values
        rate[rate <0] = 0
        tmp_array[i, pos] = rate
        tmp_array[i, np.max(pos):] = -1

    # set default font size to 14
    plt.rcParams.update({'font.size': 12})
    cmap = sns.color_palette('viridis', as_cmap=True)
    cmap.set_under('w')
    g = sns.heatmap(data=tmp_array, vmin=0, vmax=1.6,cmap=cmap,  cbar_kws={'label':'Relative Methylation Rate', }, 
                    yticklabels=chrom_list,)

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
    plt.savefig(figure_fn, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close()




@click.command(help='Calculate and plot the methylation rate in 100 kb bins for the genome.')
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
    if exp != 'Live':
        raise ValueError('exp must be Live for rate calculation, found %s' %exp)
    sample_fn = config['sample']['sample_sheet']
    bw_t2t_prefix = config['bigwig']['bw_t2t_prefix']
    sample_list, data_points = load_sample_sheet(sample_fn, exp)
    cen_loc_fn = resource['general']['cen_loc_table']

    figure_dir = config['output']['figure_dir']
    figure_source_dir = os.path.join(figure_dir, 'source')
    os.makedirs(figure_source_dir, exist_ok=True)

    figure_prefix = config['output']['figure_prefix']
    figure_dir = os.path.join(figure_dir, 'rate')
    os.makedirs(figure_dir, exist_ok=True)

    figure_fn = os.path.join(figure_dir, f'{figure_prefix}.GenomeMap.100kb.png')

    figure_source_fn = os.path.join(figure_source_dir, f'{figure_prefix}.GenomeMap.100kb.csv')
    figure_source_data_db = os.path.join(figure_source_dir, f'{figure_prefix}.GenomeMap.100kb.raw.pkl')

    if run in ['full', 'calc']:
        bw_t2t_dict = search_bigwig(sample_list, bw_t2t_prefix)
        calc_genome_map_rate(bw_t2t_dict, sample_list, data_points, figure_source_fn, figure_source_data_db)
        logger.info('Calulcation of Genome Methylation Rate is done!')

    if run in ['full', 'plot']:
        if not os.path.isfile(figure_source_fn):
            raise OSError('Cannot find %s, please calculate rate again' %figure_source_fn)
        genome_rate_pd = pd.read_csv(figure_source_fn, index_col=False)
        cen_pos_pd = pd.read_csv(cen_loc_fn, index_col=False)
        
        plot_genome_map(genome_rate_pd, cen_pos_pd, figure_fn)
        logger.info('Figure of Genome Methylation Rate is plotted!')        


if __name__ == '__main__':
    main()







         

        


