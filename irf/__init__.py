import numpy as np


def theta_degrees_to_theta_squared_mm(theta):
    '''
    Convert theta from fact-tols output (in degrees) to theta^2 in mm.
    This formula contains at two approximations.
    1. The mirror is a perfect parabola
    2. The area around the point source in the camera, aka the 'phase space',  grows
        with r**2. I think its missing a cosine somewhere. but thats okay.
    '''
    pixelsize = 9.5  # mm
    fov_per_pixel = 0.11  # degree
    return (theta * (fov_per_pixel / pixelsize))**2


def histograms(
        predictions,
        showers,
        theta_square_cut,
        prediction_threshold,
        energy_bins,
        zenith_bins,
        ):
    '''
    calculate the matrices from the analysed and the simulated events.
    when dividing these matrices you get the some response which,
    when normalised correctly, corresponds to the collection area.

    returns hist_showers, hist_data,  x_edges, y_edges
    '''
    # apply cuts
    # print('Selecting data with prediction_threshold {} and theta_square_cut {}'.format(
    #     prediction_threshold, theta_square_cut))

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
    # print('{} gammas left after applying cuts'.format(len(predictions)))

    showers['energy'] = showers['energy'].apply(np.log)
    showers['zenith'] = showers['zenith'].apply(np.rad2deg)

    predictions['energy'] = predictions['MCorsikaEvtHeader.fTotalEnergy'].apply(np.log10)
    predictions['zenith'] = predictions['MCorsikaEvtHeader.fZd'].apply(np.rad2deg)

    hist_showers, x_edges, y_edges = np.histogram2d(
        showers.energy, showers.zenith, bins=(energy_bins, zenith_bins)
    )
    hist_data, x_edges, y_edges = np.histogram2d(
        predictions.energy, predictions.zenith, bins=(energy_bins, zenith_bins)
    )

    return hist_showers, hist_data,  x_edges, y_edges
