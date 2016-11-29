import numpy as np
import click
import pandas as pd
from IPython import embed
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
# plt.style('ggplot')
plt.style.use('ggplot')

def theta_degrees_to_theta_squared_mm(theta):
    pixelsize = 9.5 #mm
    fov_per_pixel = 0.11 #degree
    return (theta*(fov_per_pixel/pixelsize))**2

def histograms(predictions, showers, theta_square_cut, prediction_threshold):

    # apply cuts
    print('Selecting data with prediction_threshold {} and theta_square_cut {}'.format(prediction_threshold, theta_square_cut))
    predictions['signal_theta_square'] = theta_degrees_to_theta_squared_mm(predictions['signal_theta'])

    predictions = predictions.copy().query('signal_prediction >= {} & signal_theta_square < {}'.format(prediction_threshold, theta_square_cut))
    print('{} gammas left after applying cuts'.format(len(predictions)))

    showers['energy'] = showers['energy'].apply(np.log)
    showers['zenith'] = showers['zenith'].apply(np.rad2deg)

    predictions['energy'] = predictions['MCorsikaEvtHeader.fTotalEnergy'].apply(np.log)
    predictions['zenith'] = predictions['MCorsikaEvtHeader.fZd'].apply(np.rad2deg)

    hist_showers, x_edges, y_edges = np.histogram2d(showers.energy, showers.zenith, bins=(11, 4))
    hist_data, x_edges, y_edges = np.histogram2d(predictions.energy , predictions.zenith, bins=(11, 4))


    return hist_showers, hist_data,  x_edges, y_edges

@click.command()
@click.argument('showers', type=click.Path(exists=True, dir_okay=False, file_okay=True) )
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False, file_okay=True) )
@click.option('-c', '--prediction_threshold', type=click.FLOAT, default=0.9)
@click.option('-t', '--theta_square_cut', type=click.FLOAT, default=0.02 )
@click.option('-i', '--impact', type=click.FLOAT, default=270.0 )
def main(showers, predictions, prediction_threshold, theta_square_cut, impact):
    showers = pd.read_hdf(showers, key='table')
    predictions = pd.read_hdf(predictions, key='table')

    hist_showers, hist_data, x_edges, y_edges = histograms(predictions, showers, theta_square_cut, prediction_threshold)

    area = hist_data/hist_showers * np.pi * impact**2

    plt.figure()
    plt.title('Simulated Showers')
    plt.imshow(hist_showers.T, interpolation='nearest', origin='lower',cmap='plasma', extent=(showers.energy.min(), showers.energy.max(), 0, 60), aspect='auto')
    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel('Zenith Angle')
    plt.colorbar()
    plt.savefig('showers.pdf')

    plt.figure()
    plt.title('Events after Separation')
    plt.suptitle('Predictions Threshold > {} and Theta Square < {}'.format(prediction_threshold, theta_square_cut))
    plt.imshow(hist_data.T, interpolation='nearest', origin='lower',cmap='plasma', extent=(showers.energy.min(), showers.energy.max(), 0, 60), aspect='auto')
    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel('Zenith Angle')
    plt.colorbar()
    plt.savefig('events.pdf')


    plt.figure()
    plt.title('Collection Area')
    plt.suptitle('Predictions Threshold > {} and Theta Square < {}'.format(prediction_threshold, theta_square_cut))
    plt.imshow(area.T, interpolation='nearest', origin='lower',cmap='plasma', extent=(showers.energy.min(), showers.energy.max(), 0, 60), aspect='auto')
    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel('Zenith Angle')
    plt.colorbar()
    plt.savefig('hist.pdf')

    plt.figure()

    bin_width = x_edges[1] - x_edges[0]
    bin_center = x_edges[1:] - 0.5 * bin_width

    for i in range(area.shape[1]):
        plt.errorbar(bin_center, area[:, i], xerr=bin_width/2.0, linestyle='', label='{:2.0f} - {:2.0f}'.format(y_edges[i], y_edges[i+1]))

    plt.title('Collection Area')
    plt.suptitle('Predictions Threshold > {} and Theta Square < {}'.format(prediction_threshold, theta_square_cut))
    plt.legend(loc='upper left')
    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel(r'$\mathrm{Area} / m^2$')
    plt.savefig('area.pdf')
if __name__ == '__main__':
    main()
