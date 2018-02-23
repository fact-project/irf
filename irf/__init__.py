from .collection_area import collection_area, collection_area_to_irf_table

import numpy as np
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator


__all__ = ['collection_area', 'collection_area_to_irf_table']



class IRF():

    def __init__(self, grid, data):
        self.grid = grid
        self.data = data
        self.interpolator = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=None)

    def __repr__(self):
        us = [l.unit for l in self.grid]
        s = f'IRF containing data of shape {self.data.shape} with unit {self.data.unit}. \n'
        s += f'Evaluation nodes for the interpolator have units {us} '
        return s


class EffectiveArea(IRF):

    @u.quantity_input(energy=u.TeV, theta=u.deg)
    def evaluate(self, energy, theta):
        e = energy.to(self.grid[0].unit).value
        t = theta.to(self.grid[1].unit).value

        aeff = self.interpolator((e, t))
        aeff[aeff < 0] = 0
        return aeff * self.data.unit

    @property
    def max_energy(self):
        return self.grid[0].max()

    @property
    def min_energy(self):
        return self.grid[0].min()

    @property
    def max_offset(self):
        return self.grid[1].max()

    @property
    def min_offset(self):
        return self.grid[1].min()


class EnergyDispersion(IRF):


    @u.quantity_input(e_true=u.TeV, theta=u.deg)
    def evaluate(self, e_true, migra, theta):
        et = e_true.to(self.grid[0].unit).value
        t = theta.to(self.grid[2].unit).value
        edisp = self.interpolator((et, migra, t))

        # weirdly enough this happens
        if self.data.unit == '':
            return edisp
        print(self.data.unit)
        return edisp * self.data.unit



    @u.quantity_input(e_true=u.TeV, theta=u.deg)
    def response(self, e_true, theta=0.5 * u.deg):
        mu = self.grid[1]
        p_mu = self.evaluate(e_true=e_true, migra=mu, theta=theta)
        return np.cumsum(p_mu) / np.sum(p_mu)



def irf_from_table(table, colnames=None, interpolation_modes={}):
    # read table data that stores n-dimensional data in ogip convention
    # I haven't found any documents describing that standard. Im  sure its out there somewhere
    bounds = table.colnames[:-1]
    low_bounds = bounds[::2]
    high_bounds = bounds[1::2]

    data = table[table.colnames[-1]].quantity[0].T

    # we need to check this unit specifically because ctools writes weird units into fits tables
    # and astropy goes haywire. Juergen promised to fix this in the next release of the CTA irfs
    if data.unit == '1/s/MeV/sr':
        import astropy.units as u
        data = data.value * u.Unit('1/(s MeV sr)')


    if not colnames:
        colnames = [n.replace('_LO', '') for n in low_bounds]

    grid = []
    for colname_low, colname_high, name in zip(low_bounds, high_bounds, colnames):
        bins_low = np.ravel(table[colname_low]).quantity
        bins_high = np.ravel(table[colname_high]).quantity

        mode = interpolation_modes.get(name, 'linear')

        if mode == 'linear':
            nodes = (bins_low + bins_high) / 2
        elif mode == 'log':
            nodes = np.sqrt(bins_low * bins_high)
        grid.append(nodes)

    return grid, data
