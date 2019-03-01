from .collection_area import collection_area, collection_area_to_irf_table
from .energy_dispersion import energy_dispersion, energy_dispersion_to_irf_table, energy_migration
from .point_spread_function import point_spread_function, psf_vs_energy, psf_to_irf_table
from .exposure_map import estimate_exposure_time, build_exposure_map
import astropy.units as u
import numpy as np

__all__ = [
    'collection_area',
    'collection_area_to_irf_table',
    'energy_dispersion',
    'energy_dispersion_to_irf_table',
    'energy_migration',
    'point_spread_function',
    'psf_vs_energy',
    'psf_to_irf_table',
    'estimate_exposure_time',
    'build_exposure_map'
]


@u.quantity_input(energies=u.TeV, e_min=u.TeV, e_max=u.TeV)
def make_energy_bins(
        energies=None,
        e_min=None,
        e_max=None,
        bins=10,
        centering='linear',
):
    if energies is not None and len(energies) >= 2:
        e_min = min(energies)
        e_max = max(energies)

    unit = e_min.unit

    low = np.log10(e_min.value)
    high = np.log10(e_max.value)
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1) * unit

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths

