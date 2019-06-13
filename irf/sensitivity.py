import numpy as np
import astropy.units as u
from fact.analysis import li_ma_significance
from scipy.optimize import newton
import warnings
import pandas as pd
from scipy.stats import norm


@u.quantity_input(t_obs=u.hour, t_ref=u.hour)
def relative_sensitivity(
    n_on,
    n_off,
    alpha,
    t_obs,
    t_ref=u.Quantity(50, u.hour),
    target_significance=5,
    significance_function=li_ma_significance,
    initial_guess=0.5,
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
    initial_guess: float
        Initial guess for the root finder
    '''

    ratio = (t_ref / t_obs).si
    n_on = n_on * ratio
    n_off = n_off * ratio

    n_background = n_off * alpha
    n_signal = n_on - n_background

    if np.isnan(n_on) or np.isnan(n_off):
        return np.nan

    if n_on == 0 or n_off == 0:
        return np.nan

    if n_signal <= 0:
        return np.nan

    def equation(relative_flux):
        n_on = n_signal * relative_flux + n_background
        return significance_function(n_on, n_off, alpha) - target_significance

    try:
        result = newton(
            equation,
            x0=initial_guess,
        )
    except RuntimeError:
        warnings.warn('Could not calculate relative significance, returning nan')
        return np.nan

    return result


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


@u.quantity_input()
def calculate_sensitivity(
    df,
    theta2_cut,
    e_min: u.TeV,
    e_max: u.TeV,
    n_bins,
    t_obs: u.hour,
    t_ref: u.hour,
    n_bootstrap=100,
    min_events=5,
):
    theta_cut = np.sqrt(theta2_cut)

    def select_on(df):
        return df.theta_deg <= theta_cut

    def select_off(df):
        m = df['theta_deg_off_1'] <= theta_cut
        for i in range(2, 6):
            m |= df[f'theta_deg_off_{i}'] <= theta_cut
        return m

    df = df.copy()
    if 'weight' not in df.columns:
        df['weight'] = 1

    bin_edges = np.logspace(np.log10(e_min / u.GeV), np.log10(e_max / u.GeV), n_bins + 1)
    bin_edges = np.append(-np.inf, np.append(bin_edges, np.inf))
    bin_id = np.arange(n_bins + 2) + 1

    df['bin'] = np.digitize(df['gamma_energy_prediction'].values, bin_edges)

    sensitivity = pd.DataFrame(index=bin_id)
    sensitivity['e_low'] = bin_edges[:-1]
    sensitivity['e_high'] = bin_edges[1:]
    sensitivity['e_center'] = 0.5 * (sensitivity['e_low'] + sensitivity['e_high'])
    sensitivity['e_width'] = np.diff(bin_edges)
    sensitivity['n_on_weighted'] = 0
    sensitivity['n_off_weighted'] = 0
    sensitivity['n_on'] = 0
    sensitivity['n_off'] = 0
    sensitivity.index.name = 'bin'

    sensitivities = []
    for i in range(n_bootstrap):
        cur_sensitivity = sensitivity.copy()

        sampled = df.sample(len(df), replace=True)
        for bin_id, g in sampled.groupby('bin'):
            on = select_on(g)
            off = select_off(g)

            cur_sensitivity.loc[bin_id, 'n_on_weighted'] = g.loc[on, 'weight'].sum()
            cur_sensitivity.loc[bin_id, 'n_off_weighted'] += g.loc[off, 'weight'].sum()
            cur_sensitivity.loc[bin_id, 'n_on'] = on.sum()
            cur_sensitivity.loc[bin_id, 'n_off'] = off.sum()

        cur_sensitivity['relative_sensitivity'] = relative_sensitivity(
            cur_sensitivity['n_on_weighted'],
            cur_sensitivity['n_off_weighted'],
            alpha=0.2,
            t_obs=t_obs,
        )
        cur_sensitivity['iteration'] = i

        sensitivities.append(cur_sensitivity.reset_index())

    sensitivities = pd.concat(sensitivities)

    # aggregate bootstrap samples
    grouped = sensitivities.groupby('bin')
    keys = ['n_on', 'n_off', 'n_on_weighted', 'n_off_weighted', 'relative_sensitivity']
    for key in keys:
        sensitivity[key] = grouped[key].median()
        sensitivity[key + '_uncertainty_low'] = grouped[key].quantile(norm.cdf(-1))
        sensitivity[key + '_uncertainty_high'] = grouped[key].quantile(norm.cdf(1))
    sensitivity['count'] = grouped['relative_sensitivity'].count()

    invalid = (
        (sensitivity['n_on'] < min_events)
        | (sensitivity['n_off'] < min_events)
        | (sensitivity['count'] / n_bootstrap <= 0.95)
    )
    sensitivity.loc[invalid, 'relative_sensitivity'] = np.nan

    return sensitivity
