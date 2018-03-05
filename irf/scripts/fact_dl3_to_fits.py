import fact.io
import astropy.units as u
import click
from astropy.io import fits
from pandas.api.types import is_datetime64_any_dtype
import pandas as pd
from astropy.table import Table
from astropy.time import Time

MJDREF = 55835  # MJD sometime near FACT's first light. 2011-10-01T00:00:00 UTC


@click.command()
@click.argument(
    'input_path',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'output_path',
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
)
@click.option(
    '-t', '--threshold',
    default=0.85,
    help='Prediction threshold to apply.',
)
def main(input_path, output_path, threshold):
    dl3 = fact.io.read_h5py(input_path, key='events')
    dl3 = dl3.query(f'gamma_prediction >= {threshold}').copy()

    primary_hdu = create_primary_hdu()
    event_hdu = dl3_hdu_from_pandas(dl3)

    hdu_header = event_hdu.header
    hdu_header['MJDREFI'] = MJDREF
    hdu_header['MJDREF'] = 0
    hdu_header['TIMEREF'] = 'local'
    hdu_header['TIMESYS'] = 'utc'
    hdu_header['GEOLON'] = -17.890701
    hdu_header['GEOLAT'] = 28.761795
    hdu_header['ALTITUDE'] = 2199.4
    hdu_header['EUNIT'] = 'TeV'
    hdu_header['EXTNAME'] = 'EVENTS'
    hdu_header['PRED_THR'] = threshold

    hdulist = fits.HDUList([primary_hdu, event_hdu])
    hdulist.writeto(output_path, overwrite=True)


def create_primary_hdu():
    header = fits.Header()

    header['OBSERVER'] = 'The non-insane FACT guys '
    header['COMMENT'] = 'FACT EventList. Very preliminary'
    header['COMMENT'] = 'See https://gamma-astro-data-formats.readthedocs.io/en/latest/'
    header['COMMENT'] = 'This file was created by https://github.com/fact-project/irf'
    header['COMMENT'] = 'See our full analysis here https://github.com/fact-project/open_crab_sample_analysis'

    now = Time.now().iso
    header['COMMENT'] = f'This file created {now}'

    return fits.PrimaryHDU(header=header)




def dl3_hdu_from_pandas(dl3):
    # the format expects to see some event id. I just take the index
    event_id = dl3.index

    # convert from hour angles to deg
    ra = dl3.ra_prediction.values * u.hourangle
    ra = ra.to('deg')

    dec = dl3.dec_prediction.values * u.deg

    # convert to TeV energy
    energy = dl3.gamma_energy_prediction.values * u.GeV
    energy = energy.to('TeV')

    # convert to timestamps. older pyfact versions don't do this automagically
    if not is_datetime64_any_dtype(dl3.timestamp):
        dl3.timestamp = pd.to_datetime(dl3.timestamp, infer_datetime_format=True)

    time = Time(dl3.timestamp.asobject, location=fact.instrument.constants.LOCATION)
    time = time - Time(MJDREF, scale='utc', format='mjd')

    t = Table({'EVENT_ID': event_id, 'ENERGY': energy, 'DEC': dec, 'RA': ra, 'TIME': time.to('s')})
    return fits.table_to_hdu(t)





if __name__ == '__main__':
    main()
