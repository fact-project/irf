import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from astroquery.skyview import SkyView
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import fact.io as fio
import numpy as np
import pandas as pd
from scipy.stats import expon


def estimate_exposure_time(timestamps):
    '''
    Takes numpy datetime64[ns] timestamps and returns an estimates of the exposure time in seconds
    '''
    delta_s = np.diff(timestamps).astype(int) * float(1e-9)
    # take only events that came within 30 seconds or so
    delta_s = delta_s[delta_s < 30]
    scale = delta_s.mean()

    exposure_time = len(delta_s) * scale
    loc = min(delta_s)

    # this percentile is somewhat abritrary but seems to work well.
    live_time_fraction = 1 - expon.ppf(0.1, loc=loc, scale=scale)

    return (exposure_time * live_time_fraction) * u.s


@u.quantity_input(ra=u.hourangle, dec=u.deg, fov=u.deg)
def build_exposure_regions(pointing_coords, fov=4.5 * u.deg):
    unique_pointing_positions = SkyCoord(
        ra=np.unique(pointing_coords.ra),
        dec=np.unique(pointing_coords.dec)
    )

    regions = [CircleSkyRegion(
        center=pointing,
        radius=Angle(fov) / 2
    ) for pointing in unique_pointing_positions]

    return unique_pointing_positions, regions


@u.quantity_input(event_ra=u.hourangle, event_dec=u.deg, fov=u.deg)
def build_exposure_map(pointing_coords, event_time, fov=4.5 * u.deg):
    unique_pointing_positions, regions = build_exposure_regions(pointing_coords)

    img_center = SkyCoord(ra=pointing_coords.ra.mean(), dec=pointing_coords.dec.mean())
    print('getting image')
    hdu = SkyView.get_images(position=img_center, pixels=1000, survey=['DSS'], width=2 * fov, height=2 * fov)[0][0]
    img = hdu.data
    wcs = WCS(hdu.header)

    print('starting loop')
    times = []
    for p in unique_pointing_positions:
        m = (pointing_coords.ra == p.ra) & (pointing_coords.dec == p.dec)
        exposure_time = estimate_exposure_time(event_time[m])
        print(exposure_time.to('h'))
        times.append(exposure_time)

    print(sum(times).to('h'))

    # import IPython; IPython.embed()
    masks = [r.to_pixel(wcs).to_mask().to_image(img.shape) for r in regions]
    #
    cutout = sum(masks).astype(bool)
    mask = sum([w * m for w, m in zip(times, masks)])

    return np.ma.masked_array(mask.to('h'), mask=~cutout), wcs, img


def plot_exposure(img, mask, wcs):
    ax = plt.subplot(projection=wcs)

    ax.imshow(img, cmap='gray')
    d = ax.imshow(mask, alpha=0.7)
    cb = plt.colorbar(d)
    cb.set_label('Live Time / Hours')
    ax.set_xlabel('Galactic Longitude')
    ax.set_ylabel('Galactic Latitude')

    crab = SkyCoord.from_name('Crab Nebula')
    ax.scatter(crab.ra.deg, crab.dec.deg, transform=ax.get_transform('icrs'), label='Crab Nebula')
    ax.legend()


if __name__ == '__main__':
    runs = fio.read_data('crab_dl3.hdf5', key='runs')
    dl3 = fio.read_data('crab_dl3.hdf5', key='events')

    data = pd.merge(runs, dl3, on=['run_id', 'night'] )

    timestamps = pd.to_datetime(data.timestamp).values
    total_ontime = estimate_exposure_time(timestamps)
    print('total exposure time: {}'.format(total_ontime.to(u.h)))

    ra = data.ra_prediction.values * u.hourangle
    dec = data.dec_prediction.values * u.deg

    ra_pointing = data.right_ascension.values * u.hourangle
    dec_pointing = data.declination.values * u.deg

    event_coords = SkyCoord(ra=ra, dec=dec)
    pointing_coords = SkyCoord(ra=ra_pointing, dec=dec_pointing)

    mask, wcs, img = build_exposure_map(pointing_coords, timestamps)

    ax = plot_exposure(img, mask, wcs)
    plt.savefig('exposure.pdf')
    # plt.show()
