import numpy as np
import click
from fact.io import read_data
from irf import collection_area, energy_dispersion, energy_migration
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from astropy.coordinates.angle_utilities import angular_separation

columns_to_read = [
    'corsika_event_header_num_reuse',
    'corsika_event_header_event_number',
    'corsika_run_header_run_number',
    'theta_deg',
    'corsika_event_header_total_energy',
    'gamma_prediction',
    'gamma_energy_prediction',
    'aux_pointing_position_zd',
    'aux_pointing_position_az',
    'source_position_az',
    'source_position_zd',
]


def fov_offset(df):
    pointing_lat = (90 - df.aux_pointing_position_zd.values) * u.deg
    pointing_lon = df.aux_pointing_position_az.values * u.deg

    source_lat = (90 - df.source_position_zd.values) * u.deg
    source_lon = df.source_position_az.values * u.deg

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat)


@click.command()
@click.argument(
    'showers', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument(
    'predictions', type=click.Path(
        exists=True,
        dir_okay=False,
    ))
@click.option(
    '-o', '--outputfile', type=click.Path(
        exists=False,
        dir_okay=False,
    ))
@click.option('-n', '--n_aeff', type=click.INT, default=20, help='number of bins to use for effective area')
@click.option('-e', '--n_edisp', type=click.INT, default=60, help='number of bins to use for edisp')
@click.option('-c', '--prediction_threshold', type=click.FLOAT, default=0.85)
@click.option('-t', '--theta_square_cut', type=click.FLOAT, default=0.02)
@click.option('-i', '--impact', type=click.FLOAT, default=270.0)
def main(
        showers,
        predictions,
        outputfile,
        n_aeff,
        n_edisp,
        prediction_threshold,
        theta_square_cut,
        impact,
):

    showers = read_data(showers, key='showers')

    predictions = read_data(predictions, key='events', columns=columns_to_read)

    q = f'gamma_prediction >= {prediction_threshold} & theta_deg <= {np.sqrt(theta_square_cut)}'
    selected_events = predictions.query(q)

    shower_energy = (showers.energy.values * u.GeV).to('TeV')
    true_event_energy = (selected_events.corsika_event_header_total_energy.values * u.GeV).to('TeV')
    predicted_event_energy = (selected_events.gamma_energy_prediction.values * u.GeV).to('TeV')
    offset = fov_offset(selected_events).to('deg').value

    low = np.log10(shower_energy.min().value)
    high = np.log10(shower_energy.max().value)
    bin_edges = np.logspace(low, high, endpoint=True, num=n_aeff + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 10))


    r = collection_area(
        shower_energy,
        true_event_energy,
        impact=impact * u.m,
        bins=bin_edges,
        sample_fraction=1,
    )

    area, bin_center, bin_width, lower_conf, upper_conf = r

    # matplotlib wants relative offsets for errors. the conf values are absolute.
    lower = area - lower_conf
    upper = upper_conf - area
    ax1.errorbar(
        bin_center,
        area.value,
        xerr=bin_width / 2.0,
        yerr=[lower.value, upper.value],
        linestyle='',
    )
    ax1.set_xscale('log')

    # plt.legend(loc='upper left')
    ax1.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax1.set_ylabel(r'$\mathrm{Mean Effective\; Area} / \mathrm{m}^2$')

    fov = 4.5
    areas = []
    for lower, upper in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5)]:
        m = (offset > lower) & (offset < upper)
        f = (upper**2 - lower**2) / ((fov / 2) ** 2)

        r = collection_area(
            shower_energy,
            true_event_energy[m],
            impact=impact * u.m,
            bins=bin_edges,
            sample_fraction=f,
        )

        area, bin_center, bin_width, lower_conf, upper_conf = r
        areas.append(area)
        # matplotlib wants relative offsets for errors. the conf values are absolute.

        lower = area - lower_conf
        upper = upper_conf - area
        ax2.errorbar(
            bin_center,
            area.value,
            xerr=bin_width / 2.0,
            yerr=[lower.value, upper.value],
            linestyle='',
        )


    mean_area = np.array(areas).mean(axis=0)
    ax2.errorbar(
        bin_center,
        mean_area,
        xerr=bin_width / 2.0,
        linestyle='',
        color='black',
        label='mean'
    )

    ax2.set_xscale('log')

    # plt.legend(loc='upper left')
    ax2.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax2.set_ylabel(r'$\mathrm{Effective\; Area} / \mathrm{m}^2$')







    hist, bins_e_true, bins_e_prediction = energy_dispersion(true_event_energy, predicted_event_energy, bins=n_edisp)

    ax3.pcolormesh(bins_e_true, bins_e_prediction, hist, cmap='GnBu')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax3.set_ylabel(r'$E_{\mathrm{Reco}} /  \mathrm{TeV}$')

    hist, bins_e_true, bins_mu = energy_migration(true_event_energy, predicted_event_energy, bins=n_edisp)

    ax4.pcolormesh(bins_e_true, bins_mu, hist, cmap='GnBu', norm=PowerNorm(0.5))
    ax4.set_xscale('log')
    ax4.set_ylabel(r'$E_{\mathrm{Reco}} / E_\mathrm{{True}}$')
    ax4.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax4.set_ylim([0, 3])


    if outputfile:
        plt.savefig(outputfile)
    else:
        plt.show()



if __name__ == '__main__':
    main()
