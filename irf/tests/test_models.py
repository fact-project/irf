from irf.models import Spectrum, MCSpectrum, CrabSpectrum
import numpy as np
import astropy.units as u


def test_extended_source():
    n = 9.6e-9 / (u.GeV * u.cm**2 * u.s * u.sr)
    spectrum = Spectrum(index=-2, normalization_constant=n)

    assert spectrum.extended_source == True

    n = 5.0e-4 / (u.GeV * u.km**2 * u.h)
    spectrum = Spectrum(index=-2, normalization_constant=n)

    assert spectrum.extended_source == False


def test_reweighing():
    '''
    Create an MCSpectrum and sample some energies from it. Then reweight these
    samples to a Crab spectrum.

    '''

    e_min, e_max = 0.003 * u.TeV , 300 * u.TeV
    t_assumed_obs = 50 * u.h

    energy_bins = np.logspace(-2, 2, num=20) * u.TeV


    mc = MCSpectrum(
        e_min=e_min,
        e_max=e_max,
        total_showers_simulated=100000,
        generation_area=1 * u.km**2,
        index=-2.0
    )
    random_energies = mc.draw_energy_distribution(500000)

    crab = CrabSpectrum()


    w = mc.reweigh_to_other_spectrum(crab, random_energies, t_assumed_obs=t_assumed_obs)
    expected_mc_events, _ = np.histogram(
        random_energies,
        bins=energy_bins,
        weights=w,
    )

    expected_events_from_crab = crab.expected_events_for_bins(
        energy_bins=energy_bins,
        area=1000 * u.m**2,
        t_obs=t_assumed_obs,
    )

    np.allclose(expected_mc_events, expected_events_from_crab)
