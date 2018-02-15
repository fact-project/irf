from irf import estimate_exposure_time, build_exposure_map
import click
import fact.io as fio
import pandas as pd
from astroquery.skyview import SkyView
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch
import astropy.units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt


@u.quantity_input(fov=u.deg)
def get_sdss_sky_image(img_center, fov=9 * u.deg, n_pix=1000):
    '''
    A small helper method which uses astroquery to get an image from the sdss.
    This requires internet access and fails pretty often due to http timeouts.
    '''
    hdu = SkyView.get_images(
        position=img_center,
        pixels=n_pix,
        survey=['DSS'],
        width=fov,
        height=fov)[0][0]

    img = hdu.data
    wcs = WCS(hdu.header)
    return img, wcs


@click.command()
@click.argument(
    'dl3_path',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'output_path',
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
)
@click.option(
    '-n',
    '--n_pix',
    default=1000,
    help='number of pixels along the edge of the produced image',
)
@click.option(
    '-s',
    '--source_name',
    default=None,
    help=
    'If supplied, e.g. "Crab Nebula" will draw the position of that source into the image',
)
@click.option(
    '--background/--no-background',
    default=True,
    help='If true, downloads SDSS image for backgrund image in the plot')
def main(dl3_path, output_path, n_pix, source_name, background):
    '''
    Takes FACT dl3 output and plots a skymap which is being saved to the output_path.
    '''
    runs = fio.read_data(dl3_path, key='runs')
    dl3 = fio.read_data(dl3_path, key='events')

    data = pd.merge(runs, dl3, on=['run_id', 'night'])

    timestamps = pd.to_datetime(data.timestamp).values
    total_ontime = estimate_exposure_time(timestamps)
    print('Total estimated exposure time: {}'.format(total_ontime.to(u.h)))

    ra_pointing = data.right_ascension.values * u.hourangle
    dec_pointing = data.declination.values * u.deg
    pointing = SkyCoord(ra=ra_pointing, dec=dec_pointing)

    img = None
    wcs = None
    if background:
        img_center = SkyCoord(ra=pointing.ra.mean(), dec=pointing.dec.mean())
        img, wcs = get_sdss_sky_image(
            img_center=img_center, n_pix=n_pix, fov=9 * u.deg)

    mask, wcs = build_exposure_map(pointing, timestamps, shape=(n_pix, n_pix))

    ax = plot_exposure(mask, wcs, image=img)
    if source_name:
        source = SkyCoord.from_name('Crab Nebula')
        ax.scatter(
            source.ra.deg,
            source.dec.deg,
            transform=ax.get_transform('icrs'),
            label=source_name,
            s=10**2,
            facecolors='none',
            edgecolors='r',
        )
        ax.legend()

    plt.savefig(output_path, dpi=200)


def plot_exposure(mask, wcs, image=None):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)

    if image is not None:
        norm = ImageNormalize(
            image,
            interval=ZScaleInterval(contrast=0.05),
            stretch=AsinhStretch(a=0.2))
        ax.imshow(image, cmap='gray', norm=norm, interpolation='nearest')

    d = ax.imshow(mask, alpha=0.7)
    cb = plt.colorbar(d)
    cb.set_label('Live Time / Hours')
    ax.set_xlabel('Galactic Longitude')
    ax.set_ylabel('Galactic Latitude')

    return ax


if __name__ == '__main__':
    main()
