import click
from .lazy_group import LazyGroup


@click.group(
    help='Analyze BigWig files for Methylated Fractions',
    cls=LazyGroup,
    lazy_subcommands={
        'featPercentile': 'methylFracAnalyzer.analyze_feature_percentile.main',
        'featRate': 'methylFracAnalyzer.analyze_feature_rate.main',
        'genomeFrac': 'methylFracAnalyzer.analyze_genome_frac.main',
        'genomeRate': 'methylFracAnalyzer.analyze_genome_map_rate.main',
        'phasing': 'methylFracAnalyzer.analyze_phasing.main',
    })

def cli():
    pass




if __name__ == '__main__':
    cli()
