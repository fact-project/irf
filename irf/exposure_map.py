import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy import wcs
from regions import CircleSkyRegion
import numpy as np
from scipy.stats import expon


def estimate_exposure_time(timestamps):
    '''
    Takes numpy datetime64[ns] timestamps and returns an estimates of the exposure time in seconds.
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


def _build_standard_wcs(image_center, shape, naxis=2, fov=9 * u.deg):
    width, height = shape

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [width / 2 + 0.5, height / 2 + 0.5]
    w.wcs.cdelt = np.array([-fov.value / width, fov.value / height])
    w.wcs.crval = [image_center.ra.deg, image_center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.radesys = 'FK5'
    w.wcs.equinox = 2000.0
    w.wcs.cunit = ['deg', 'deg']
    w._naxis = [width, height]
    return w


@u.quantity_input(event_ra=u.hourangle, event_dec=u.deg, fov=u.deg)
def build_exposure_map(pointing_coords, event_time, fov=4.5 * u.deg, wcs=None, shape=(1000, 1000)):

    if not wcs:
        image_center = SkyCoord(ra=pointing_coords.ra.mean(), dec=pointing_coords.dec.mean())
        wcs = _build_standard_wcs(image_center, shape, fov=2 * fov)

    unique_pointing_positions, regions = build_exposure_regions(pointing_coords)

    times = []
    for p in unique_pointing_positions:
        m = (pointing_coords.ra == p.ra) & (pointing_coords.dec == p.dec)
        exposure_time = estimate_exposure_time(event_time[m])
        times.append(exposure_time)

    masks = [r.to_pixel(wcs).to_mask().to_image(shape) for r in regions]
    cutout = sum(masks).astype(bool)
    mask = sum([w.to('h').value * m for w, m in zip(times, masks)])

    return np.ma.masked_array(mask, mask=~cutout), wcs
