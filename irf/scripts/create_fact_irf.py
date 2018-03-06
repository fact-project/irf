import fact.io
from irf import collection_area, collection_area_to_irf_table, energy_dispersion, energy_dispersion_to_irf_table
import astropy.units as u
import click
from astropy.io import fits


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
    default=5,
    type=int,
    help='How many energy bins to use for the IRF',
)
@click.option(
    '-t', '--threshold',
    default=0.85,
    help='Prediction threshold to apply.',
)
@click.option(
    '-m', '--max_scat',
    default=270,
    help='Maximum scatter radius (meter) used during simulations.',
)
def main(showers, predictions, output_path, bins, threshold, max_scat):
    showers = fact.io.read_data(showers, key='showers')
    predictions = fact.io.read_data(predictions, key='events').query(f'gamma_prediction > {threshold}')

    energy_true = predictions['corsika_evt_header_total_energy'].values * u.GeV
    energy_prediction = predictions['gamma_energy_prediction'].values * u.GeV

    r = collection_area(showers.energy, energy_true.value, bins=bins, impact=max_scat * u.m,)
    area, bin_center, bin_width, lower_conf, upper_conf = r

    collection_table = collection_area_to_irf_table(area, bin_center, bin_width)

    hist, bins_e_true, bins_e_prediction = energy_dispersion(
        energy_true,
        energy_prediction,
        n_bins=bins,
    )

    e_disp_table = energy_dispersion_to_irf_table(energy_true, energy_prediction, n_bins=bins)


    header = fits.Header()
    header['OBSERVER'] = 'The non-insane FACT guys '
    header['COMMENT'] = 'Behold a full enclosure FACT irf. Very preliminary'
    header['COMMENT'] = 'See https://gamma-astro-data-formats.readthedocs.io/en/latest/'
    header['PRED_THR'] = threshold
    header['MAX_SCAT'] = max_scat

    primary_hdu = fits.PrimaryHDU(header=header)
    collection_hdu = fits.table_to_hdu(collection_table)
    e_disp_hdu = fits.table_to_hdu(e_disp_table)

    hdulist = fits.HDUList([primary_hdu, collection_hdu, e_disp_hdu])
    hdulist.writeto(output_path, overwrite=True)


if __name__ == '__main__':
    main()
