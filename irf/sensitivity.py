import numpy as np
import astropy.units as u


@u.quantity_input(t_obs=u.hour, t_ref=u.hour)
def relative_sensitivity(
        n_on,
        n_off,
        alpha,
        t_obs,
        t_ref=50*u.hour,
        significance=5,
        ):
    '''
    Calculate the relative sensitivity defined as the flux
    relative to the reference source that is detectable with
    significance in t_ref.

    Parameters
    ----------
    n_on: int or array-like
        Number of signal-like events for the on observations
    n_off: int or array-like
        Number of signal-like events for the off observations
    alpha: float
        Scaling factor between on and off observations.
        1 / number of off regions for wobble observations.
    t_obs: astropy.units.Quantity of type time
        Total observation time
    t_ref: astropy.units.Quantity of type time
        Reference time for the detection
    significance: float
        Significance necessary for a detection
    '''
    t1 = n_off * np.log(n_off * (1 + alpha) / (n_on + n_off))
    t2 = n_on * np.log(n_on * (1 + alpha) / alpha / (n_on + n_off))
    return significance**2 / 2 * t_obs / t_ref * (t1 + t2)
