import numpy as np
import astropy.units as u
from fact.analysis import li_ma_significance
from scipy.optimize import newton
import warnings


@u.quantity_input(t_obs=u.hour, t_ref=u.hour)
def relative_sensitivity(
        n_on,
        n_off,
        alpha,
        t_obs,
        t_ref=50*u.hour,
        target_significance=5,
        significance_function=li_ma_significance,
        ):
    '''
    Calculate the relative sensitivity defined as the flux
    relative to the reference source that is detectable with
    significance in t_ref.

    Given measured `n_on` and `n_off` during a time period `t_obs`,
    we estimate the number of gamma events `n_signal` as `n_on - alpha * n_off`.
    The number of background events `n_background` is estimated as
    `n_off * alpha`.

    So we find the relative sensitivity as the scaling factor for `n_signal`
    that yields a significance of `target_significance`.


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
    significance_function: function
        A function f(n_on, n_off, alpha) -> significance in sigma
        Used to calculate the significance, default is the Li&Ma
        likelihood ratio test formula.
        Li, T-P., and Y-Q. Ma.
        "Analysis methods for results in gamma-ray astronomy."
        The Astrophysical Journal 272 (1983): 317-324.
        Formula (17)
    '''

    ratio = (t_ref / t_obs).si
    n_on = n_on * ratio
    n_off = n_off * ratio

    n_background = n_off * alpha
    n_signal = n_on - n_background

    if np.isnan(n_on) or np.isnan(n_off):
        return np.nan

    def equation(relative_flux):
        n_on = n_signal * relative_flux + n_background
        return li_ma_significance(n_on, n_off, alpha) - target_significance

    try:
        phi_rel = newton(equation, x0=1.0)
    except RuntimeError:
        warnings.warn('Could not calculate relative significance, returning nan')
        phi_rel = np.nan

    return phi_rel


relative_sensitivity = np.vectorize(
    relative_sensitivity,
    excluded=[
        't_obs',
        't_ref',
        'alpha',
        'target_significance',
        'significance_function',
    ]
)
