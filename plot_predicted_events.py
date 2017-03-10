import numpy as np
import click
import pandas as pd
from irf import theta_degrees_to_theta_squared_mm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
plt.style.use('ggplot')


@click.command()
@click.argument('predicted_events', type=click.Path(exists=True, dir_okay=False,))
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=False,))
@click.option('-n', '--n_energy', type=click.INT, default=11, help='energy bins')
@click.option('-z', '--n_zenith', type=click.INT, default=4, help='number of zenith bins')
@click.option('--log/--no-log', default=True, help='use log norm form color scale')
@click.option('--cmap', default='plasma', help='color map to use')
@click.option('-c', '--prediction_threshold', type=click.FLOAT, default=0.9)
@click.option('-t', '--theta_square_cut', type=click.FLOAT, default=0.02)
def main(
        predicted_events,
        outputfile,
        n_energy,
        n_zenith,
        log,
        cmap,
        prediction_threshold,
        theta_square_cut
        ):
    '''
    Plot the event distributions from the predicted gammas given in the
    PREDICTED_EVENTS input file.
    '''
    predictions = pd.read_hdf(predicted_events, key='table')

    predictions['signal_theta_square'] =\
        theta_degrees_to_theta_squared_mm(predictions['signal_theta'])

    predictions = predictions.copy()\
        .query(
        'signal_prediction >= {} & signal_theta_square < {}'.format
        (
            prediction_threshold,
            theta_square_cut
        )
    )

    predictions['energy'] = predictions['MCorsikaEvtHeader.fTotalEnergy'].apply(np.log10)
    predictions['zenith'] = predictions['MCorsikaEvtHeader.fZd'].apply(np.rad2deg)

    hist_data, x_edges, y_edges = np.histogram2d(
        predictions.energy, predictions.zenith, bins=(n_energy, n_zenith)
    )

    norm = LogNorm() if log else None

    plt.figure()
    # plt.title('Events after Separation')
    # plt.suptitle('Predictions Threshold > {} and Theta Square < {}'.format(
    #     prediction_threshold, theta_square_cut)
    # )
    plt.imshow(
        hist_data.T,
        interpolation='nearest',
        origin='lower',
        cmap=cmap,
        extent=(
                predictions.energy.min(),
                predictions.energy.max(),
                0,
                predictions.zenith.max()
            ),
        aspect='auto',
        norm=norm,
    )
    plt.xlabel(r'$\log_{10}(E /  \mathrm{GeV})$')
    plt.ylabel('Zenith Angle')
    plt.colorbar()
    plt.savefig(outputfile)


if __name__ == '__main__':
    main()
