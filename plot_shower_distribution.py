import numpy as np
import click
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')


@click.command()
@click.argument('showers', type=click.Path(exists=True, dir_okay=False,))
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=False,))
@click.option('-n', '--n_energy', type=click.INT, default=11, help='energy bins')
@click.option('-z', '--n_zenith', type=click.INT, default=4, help='number of zenith bins')
@click.option('--log/--no-log', default=True, help='use log norm form color scale')
@click.option('--cmap', default='plasma', help='color map to use')
def main(showers, outputfile, n_energy, n_zenith, log, cmap):
    '''
    Plot the shower distributions from the collected runheaders given in the
    SHOWERS input file.
    '''
    showers = pd.read_hdf(showers, key='table')

    showers['energy'] = showers['energy'].apply(np.log)
    showers['zenith'] = showers['zenith'].apply(np.rad2deg)

    hist_showers, x_edges, y_edges = np.histogram2d(
        showers.energy, showers.zenith, bins=(n_energy, n_zenith)
    )

    norm = LogNorm() if log else None

    plt.figure()
    plt.imshow(
        hist_showers.T,
        interpolation='nearest',
        origin='lower',
        cmap=cmap,
        extent=(showers.energy.min(), showers.energy.max(), 0, showers.zenith.max()),
        aspect='auto',
        norm=norm,
    )

    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel('Zenith Angle')
    plt.colorbar()
    plt.savefig(outputfile)


if __name__ == '__main__':
    main()
