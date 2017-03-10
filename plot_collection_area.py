import numpy as np
import click
import pandas as pd
from irf import histograms
from astropy.stats import binom_conf_interval
import astropy.units as u
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')


@click.command()
@click.argument('showers', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False,))
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=False,))
@click.option('-n', '--n_energy', type=click.INT, default=11, help='energy bins')
@click.option('-z', '--n_zenith', type=click.INT, default=4, help='number of zenith bins')
@click.option('-c', '--prediction_threshold', type=click.FLOAT, default=0.9)
@click.option('-t', '--theta_square_cut', type=click.FLOAT, default=0.02)
@click.option('-i', '--impact', type=click.FLOAT, default=270.0)
def main(
        showers,
        predictions,
        outputfile,
        n_energy,
        n_zenith,
        prediction_threshold,
        theta_square_cut,
        impact,
        ):

    showers = pd.read_hdf(showers, key='table')
    predictions = pd.read_hdf(predictions, key='table')

    hist_showers, hist_data, x_edges, y_edges = histograms(
        predictions, showers, theta_square_cut, prediction_threshold, n_energy, n_zenith
    )

    # use astropy to compute errors on that stuff
    conf = binom_conf_interval(hist_data, hist_showers)

    # scale confidences to match and split
    conf = conf * np.pi * impact**2 * (u.m * u.m)

    area = (hist_data / hist_showers * np.pi * impact**2) * (u.m * u.m)

    plt.figure()
    bin_width = x_edges[1] - x_edges[0]
    bin_center = x_edges[1:] - 0.5 * bin_width

    # plot as quare kilometer
    area = area.to('km**2')
    conf = conf.to('km**2')
    lower_conf, upper_conf = conf[0], conf[1]
    # plot area for each zenith bin
    for i in range(n_zenith):
        y = area[:, i]
        # matplotlib wants relative offsets for errors. the conf values are absolute.
        lower = y - lower_conf[:, i]
        upper = upper_conf[:, i] - y
        plt.errorbar(
                bin_center,
                y.value,
                xerr=bin_width / 2.0,
                yerr=[lower.value, upper.value],
                linestyle='',
                label='{:2.0f} - {:2.0f}'.format(y_edges[i], y_edges[i + 1])
            )
    #
    # plt.title('Collection Area')
    # plt.suptitle('Predictions Threshold > {} and Theta Square < {}'.format(
    #     prediction_threshold, theta_square_cut))
    plt.legend(loc='upper left')
    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel(r'$\mathrm{Area} / \mathrm{km}^2$')

    plt.savefig(outputfile)

    plt.figure()
    # plt.title('Collection Area')
    # plt.suptitle(
    #         'Predictions Threshold > {} and Theta Square < {}'.format(
    #             prediction_threshold,
    #             theta_square_cut
    #         )
    #     )

    plt.imshow(
            area.T,
            interpolation='nearest',
            origin='lower',
            cmap='plasma',
            extent=(
                showers.energy.min(),
                showers.energy.max(),
                0,
                showers.zenith.max()
                ),
            aspect='auto'
        )
    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel('Zenith Angle')
    plt.colorbar()

    name, ext = os.path.splitext(outputfile)
    plt.savefig(name + '_hist' + ext)


if __name__ == '__main__':
    main()
