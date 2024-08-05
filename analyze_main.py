import click
from lazy_group import LazyGroup


@click.group(
    help='Analyze BigWig files for Methylated Fractions',
    cls=LazyGroup,
    lazy_subcommands={
        'featPercentile': 'analyze_feature_percentile.main',
        'featRate': 'analyze_feature_rate.main',
        'genomeFrac': 'analyze_genome_frac.main',
        'genomeRate': 'analyze_genome_map_rate.main',
        'phasing': 'analyze_phasing.main',
    })

def cli():
    pass




if __name__ == '__main__':
    cli()
