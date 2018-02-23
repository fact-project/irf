import fact.io
from irf import collection_area, collection_area_to_irf_table
import astropy.units as u
import click


@click.command()
@click.argument(
    'showers',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'predictions',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'output_path',
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
)
@click.option(
    '-b', '--bins',
    default=-1,
    type=int,
    help='How many energy bins to use for the IRF',
)
def main(showers, predictions, output_path, bins):
    showers = fact.io.read_data(showers, key='showers')
    predictions = fact.io.read_data(predictions, key='events')

    r = collection_area(showers.energy, predictions.energy, bins=bins, impact=270*u.m,)
    area, bin_center, bin_width, lower_conf, upper_conf  = r

    table = collection_area_to_irf_table(area, bin_center, bin_width)
    # TODO also write edisp table
    table.write(output_path, overwrite=True)
