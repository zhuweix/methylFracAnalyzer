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
import statsmodels.api as sm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(message)s")
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

def load_sample_sheet(sample_fn: str, exp: str):
    sample_pd = pd.read_csv(sample_fn, sep='\t', names=['prefix', 'sample'])
    sample_list = sample_pd['sample'].values
    if exp == 'Live':
        data_points = [int(s[:-1]) for s in sample_list]
    elif exp == 'Nuclei':
        data_points = [np.round(float(s[:-1])/32,2) for s in sample_list]
    return sample_list, data_points

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
        return np.nan, np.nan, np.nan
    data_row = data_row[valid_pos]
    X = time_points
    y = data_row.reshape(-1, 1)
    if np.any(np.isnan(y)):
        return np.nan, np.nan, np.nan

    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    r2 = result.rsquared

    c, k = result.params
    return -k, c, r2


def calc_methylation_rate(data_pd, data_points):
    xpos = np.array([da for da in data_points if da > 1e-6])
    rate_pd = []
    for feat, tmp_pd in data_pd.groupby('Feature'):
        values = []
        max_val = 0
        for da in data_points:
            if da < 1e-6:
                continue
            val = tmp_pd.loc[tmp_pd['Time (h)'] == da]['q50'].values[0] / 100
            
            if val > .99:
                val = .99
                max_val += 1
            values.append(val)
        values = np.array(values)
        if max_val > 1:
            max_pos = np.where(values>=.99)[0]
            values[max_pos[1]:] = np.nan
        values = np.log(1 - np.array(values))
        k, c, r2 = calc_kinetic_rate(values, xpos)
        row = [feat, k, c, r2] + list(values)
        rate_pd.append(row)
    columns = ['Feature', 'Slope', 'Intercept', 'RSquared'] + [f'{x} h' for x in xpos]
    rate_pd = pd.DataFrame(rate_pd, columns=columns)
    
    return rate_pd


def plot_methylation_rate(data_pd, data_points, feat_list, feat_names, figure_name):
    xpos = [da for da in data_points if da > 1e-6]
    max_time = xpos[-1]
    for feat, feat_name in zip(feat_list, feat_names):
        tmp_pd = data_pd.loc[data_pd['Feature'] == feat]
        values = []
        for da in xpos:
            values.append(tmp_pd[f'{da} h'].values[0])
        k = -tmp_pd['Slope'].values[0]
        rate = tmp_pd['RelativeRate'].values[0]
        b = tmp_pd['Intercept'].values[0]
        if feat == 'Genome':
            g = sns.lineplot(x=[0, max_time], y=[b, k*max_time+b], label=f'{feat_name}({rate:.1f})', lw=1.5, color='.7')
            g = sns.scatterplot(x=xpos, y=values, marker='o', s=10, color='.7')
        else:            
            g = sns.lineplot(x=[0, max_time], y=[b, k*max_time+b], label=f'{feat_name}({rate:.1f})', lw=1.5)
            g = sns.scatterplot(x=xpos, y=values, marker='o', s=10)
    times_ = [0] + xpos
    g.set_xticks(times_)
    g.set_xlim([0, max_time+2])
    if "Centromere" in figure_name:
        g.set_ylim([-2.2, 0])
    else:
        g.set_ylim([-4.65, 0])
    g.set_xlabel('Time after transduction (h)', fontsize='14')
    g.set_ylabel('ln(1-FracMethylation)', fontsize='14')        
    g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Feature(Rel Rate)')
    plt.gcf().set_size_inches((5, 3.5))
    plt.tight_layout()
    plt.savefig(figure_name, dpi=300, facecolor='white', bbox_inches='tight', transparent=False)
    plt.close()    


@click.command(help='Calculate and plot the methylation rates using the median methylated fractions. This subcommand is dependent on the results from the feat_percentile subcommand')
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
    if exp != "Live":
        raise ValueError('exp must be Live for rate calculation, found %s' %exp)
    sample_fn = config['sample']['sample_sheet']

    _, data_points = load_sample_sheet(sample_fn, exp)

    figure_dir = config['output']['figure_dir']
    figure_source_dir = os.path.join(figure_dir, 'source')
    os.makedirs(figure_source_dir, exist_ok=True)
    figure_dir = os.path.join(figure_dir, 'rate')
    os.makedirs(figure_dir, exist_ok=True)    
    figure_prefix = config['output']['figure_prefix']

    feat_fn = os.path.join(figure_source_dir, f"{figure_prefix}.hg38.feature.quantile.csv")
    cen_fn = os.path.join(figure_source_dir, f"{figure_prefix}.t2t.centromere.quantile.csv")
    chromhmm_fn = os.path.join(figure_source_dir, f"{figure_prefix}.t2t.chromhmm.quantile.csv")

    feat_rate_fn = os.path.join(figure_source_dir, f"{figure_prefix}.hg38.feature.rate.csv")
    cen_rate_fn = os.path.join(figure_source_dir, f"{figure_prefix}.t2t.centromere.rate.csv")
    chromhmm_rate_fn = os.path.join(figure_source_dir, f"{figure_prefix}.t2t.chromhmm.rate.csv")


    if run in ['full', 'calc']:
        if not os.path.exists(feat_fn):
            raise OSError('Cannot found %s, please run featPercentile first.' % feat_fn)
        if not os.path.exists(cen_fn):
            raise OSError('Cannot found %s, please run featPercentile first.' % cen_fn)
        if not os.path.exists(chromhmm_fn):
            raise OSError('Cannot found %s, please run featPercentile first.' % chromhmm_fn)

        feat_pd = pd.read_csv(feat_fn, index_col=None)
        centromoere_pd = pd.read_csv(cen_fn, index_col=None)
        chromhmm_quantile_pd = pd.read_csv(chromhmm_fn, index_col=None)

        feat_rate_pd = calc_methylation_rate(feat_pd, data_points)
        
        genome_row = feat_rate_pd.loc[feat_rate_pd['Feature'] == 'Genome'].reset_index(drop=True)
        genome_rate = genome_row['Slope'].values[0]
        genome_row['RelativeRate'] = 1
        feat_rate_pd.insert(1, 'RelativeRate', feat_rate_pd['Slope'] / genome_rate)
        feat_rate_pd.to_csv(feat_rate_fn, index=False)

        cen_rate_pd = calc_methylation_rate(centromoere_pd, data_points)
        cen_rate_pd.insert(1, 'RelativeRate', cen_rate_pd['Slope'] / genome_rate)
        cen_rate_pd = pd.concat([cen_rate_pd, genome_row], ignore_index=True)
        cen_rate_pd.to_csv(cen_rate_fn, index=False)

        chromhmm_rate_pd = calc_methylation_rate(chromhmm_quantile_pd, data_points)
        chromhmm_rate_pd.insert(1, 'RelativeRate', chromhmm_rate_pd['Slope'] / genome_rate)
        chromhmm_rate_pd = pd.concat([chromhmm_rate_pd, genome_row], ignore_index=True)
        chromhmm_rate_pd.to_csv(chromhmm_rate_fn, index=False)

        logger.info('Rate Calculation finished')

    if run in ['full', 'plot']:
        if not os.path.exists(feat_rate_fn):
            raise OSError('Cannot found %s, please run featPercentile first.' % feat_rate_fn)
        if not os.path.exists(cen_rate_fn):
            raise OSError('Cannot found %s, please run featPercentile first.' % cen_fn)
        if not os.path.exists(chromhmm_rate_fn):
            raise OSError('Cannot found %s, please run featPercentile first.' % chromhmm_rate_fn)
        
        feat_rate_pd = pd.read_csv(feat_rate_fn, index_col=False)
        cen_rate_pd = pd.read_csv(cen_rate_fn, index_col=False)
        chromhmm_rate_pd = pd.read_csv(chromhmm_rate_fn, index_col=False)

        figure_name = f'{figure_prefix}.Gene.RelativeRate.byNDR.png'
        figure_name = os.path.join(figure_dir, figure_name)
        feat_list = ['Promoter Active','Promoter Inactive', 'Genebody Active', 'Genebody Inactive', 'Genome']

        plot_methylation_rate(feat_rate_pd, data_points, feat_list, feat_list, figure_name)

        figure_name = f'{figure_prefix}.Gene.RelativeRate.byNDR.png'
        figure_name = os.path.join(figure_dir, figure_name)
        feat_list = ['Promoter Active','Promoter Inactive', 'Genebody Active', 'Genebody Inactive', 'Genome']

        plot_methylation_rate(feat_rate_pd, data_points, feat_list, feat_list, figure_name)

        feat_list = [f'Promoter Q{i}' for i in range(5, 0, -1)] + ['Promoter NoTrans', 'Genome']
        feat_names = [f'Q{i}' for i in range(5, 0, -1)] + ['NoTrans', 'Genome']
        figure_name = f'{figure_prefix}.Gene.RelativeRate.byExpression.png'
        figure_name = os.path.join(figure_dir, figure_name)        
        plot_methylation_rate(feat_rate_pd, data_points, feat_list, feat_names, figure_name)

        feat_list = ["Promoter", "Genebody", 'tRNA', 'enhancer', 'cpgIsland', 'silencer', 'origin_of_replication', 'centromere', 'Genome']
        feat_names = ["Promoter", "Genebody", 'tRNA', 'Enhancer', 'CpgIsland', 'Silencer', 'Origin of Replication', 'Centromere', 'Genome']
        figure_name = f'{figure_prefix}.GeneralFeature.RelativeRate.png'
        figure_name = os.path.join(figure_dir, figure_name)   
        plot_methylation_rate(feat_rate_pd, data_points, feat_list, feat_names, figure_name)

        if org == 'MCF7':
            active_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2',]            
            other_list = ['ZNF/Rpts', 'Het', 'Het2', 'ReprPc', 'Biv', 'NoMark']
        else:
            active_list = ['TSS', 'TssFlnk1', 'TssFlnk2', 'Tx', 'TxWk', 'EnhG1', 'EnhG2', 'EnhA1', 'EnhA2', 'EnhWk']           
            other_list = ['ZNF/Rpts', 'Het', 'ReprPc', 'Biv', 'NoMark']

        feat_list = active_list + ['Genome']  
        figure_name = f'{figure_prefix}.ChromHMM.RelativeRate.ActiveChromatin.png'     
        figure_name = os.path.join(figure_dir, figure_name)   
        plot_methylation_rate(chromhmm_rate_pd, data_points, feat_list, feat_list, figure_name)  

        feat_list = other_list + ['Genome']  
        figure_name = f'{figure_prefix}.ChromHMM.RelativeRate.OtherChromatin.png'     
        figure_name = os.path.join(figure_dir, figure_name)   
        plot_methylation_rate(chromhmm_rate_pd, data_points, feat_list, feat_list, figure_name)           
        
        feat_list = ['αSat Active', 'αSat Inactive', 'αSat Other', 'HSat1', 
            'HSat2', 'HSat3', 'βSat', 'rDNA', 'OtherSat', 'NonSat', 'Genome']    
        figure_name = f'{figure_prefix}.Centromere.RelativeRate.CenElements.png'
        figure_name = os.path.join(figure_dir, figure_name)   
        plot_methylation_rate(cen_rate_pd, data_points, feat_list, feat_list, figure_name)     

        sf_list = ['SF1', 'SF2', 'SF3', 'SF01']
        feat_list = ['αSat Active'] + ['αSat ' + s for s in sf_list] + ['Genome']
        figure_name = f'{figure_prefix}.Centromere.RelativeRate.Active_αSat.png' 
        figure_name = os.path.join(figure_dir, figure_name)   
        plot_methylation_rate(cen_rate_pd, data_points, feat_list, feat_list, figure_name)    

        feat_list = ['CENPA'] + ['CENPA ' + s for s in sf_list] + ['Genome']
        figure_name = f'{figure_prefix}.Centromere.RelativeRate.CENPA.png'                                   
        figure_name = os.path.join(figure_dir, figure_name)   
        plot_methylation_rate(cen_rate_pd, data_points, feat_list, feat_list, figure_name) 
        
        logger.info('Rate Figures finished')



if __name__ == '__main__':
    main()

