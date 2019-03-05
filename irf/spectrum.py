import numpy as np
import astropy.units as u
from scipy.stats import norm
from scipy.integrate import quad


@u.quantity_input(energies=u.TeV, e_min=u.TeV, e_max=u.TeV)
def make_energy_bins(
        energies=None,
        e_min=None,
        e_max=None,
        bins=10,
        centering='linear',
):
    if energies is not None and len(energies) >= 2:
        e_min = min(energies)
        e_max = max(energies)

    unit = e_min.unit

    low = np.log10(e_min.value)
    high = np.log10(e_max.value)
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1) * unit

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths



class Spectrum():
    '''
    A class containing usefull methods for working with power law spectra.
    The  normalization_constant should a a unit like 1 /(TeV m^2 h)
    and the index ( e.g. -2.0) should have not unit attached.

    See the subclasses `~models.CosmicRaySpectrum` and `~models.CrabSpectrum`
    for usefull physical spectra.

    Note that for now this only works for simple power laws.
    '''


    def __init__(self, index, normalization_constant):
        self.index = index
        self.normalization_constant = normalization_constant


    @property
    def extended_source(self):
        return u.sr.si in [b.si for b in self.normalization_constant.unit.bases ]


    @u.quantity_input(e_min=u.TeV, e_max=u.TeV,)
    def draw_energy_distribution(self, e_min, e_max, size, index=None):
        '''
        Draw random energies from the spectrum.
        It is different from the scipy powerlaws because it supports negative indeces.

        Parameters
        ----------
        e_min:  Quantity
            lower energy bound
        e_max: Quantity
            upper energy bound
        size: int
            number of random values to pick.

        Returns
        -------
        An array of random numbers, with energy units (TeV) attached, of the given size.
        '''
        if not index:
            index = self.index

        a = e_min.to('TeV').value**(index + 1)
        b = e_max.to('TeV').value**(index + 1)
        r = np.random.uniform(0, 1, size)
        k = (a + (b - a) * r)
        e = k**(1. / (index + 1))
        return e * u.TeV

    @u.quantity_input(energy=u.TeV)
    def flux(self, energy):
        '''
        Returns the (differential) flux of the spectrum at the given enrgy.
        '''
        energy = energy.to('TeV')
        flux = self.normalization_constant * (energy / u.TeV)**(self.index)

        if self.extended_source:
            return flux.to(1 / (u.TeV * u.s * u.cm**2 * u.sr))
        else:
            return flux.to(1 / (u.TeV * u.s * u.cm**2))

    @u.quantity_input(e_min=u.TeV, e_max=u.TeV, area=u.m**2, t_obs=u.s, solid_angle=u.sr)
    def expected_events(self, e_min, e_max, area, t_obs, solid_angle=None):
        '''
        Get the number of events which are expected to arrive from this spectral source.
        So its basically the integral of the flux within the given energy bounds.

        Parameters
        ----------
        e_min:  Quantity
            lower energy bound
        e_max: Quantity
            upper energy bound
        area: Quantity
            area over which particles are counted
        t_obs: Quantity
            observation time over which is being integrated
        solid_angle: Quantity (optional)
            the solid angle from which events are detected.
            Not needed for non-extended sources.
        '''
        events = self._integral(e_min, e_max) * area * t_obs

        if self.extended_source:
            if solid_angle is None:
                raise ValueError('solid angle needs to be specified for extended sources')

            events = events * solid_angle

        # at this point the value should have no units left
        assert events.si.unit.is_unity() == True
        return events.si.value


    def _integral(self, e_min, e_max):
        a = e_min.to('TeV') / u.TeV
        b = e_max.to('TeV') / u.TeV

        index = self.index

        if self.extended_source:
            N = self.normalization_constant.to(1 / (u.TeV * u.s * u.m**2 * u.sr)) * u.TeV
        else:
            N = self.normalization_constant.to(1 / (u.TeV * u.s * u.m**2)) * u.TeV

        return N * (1 / (index + 1)) * (b**(index + 1) - a**(index + 1))


    @u.quantity_input(energy_bins=u.TeV, area=u.m**2, t_obs=u.s)
    def expected_events_for_bins(
            self,
            area,
            t_obs,
            energy_bins,
            solid_angle=None,
    ):
        '''
        Get the number of events which are expected to arrive from this spectral source.
        For each of the requested bins.
        Parameters
        ----------
        area: Quantity
            area over which particles are counted
        t_obs: Quantity
            observation time over which is being integrated
        energy_bins: array like energy Quantity
            The energy binning to use.
        solid_angle: Quantity (optional)
            the solid angle from which events are detected.
            Not needed for non-extended sources.
        '''

        edges = energy_bins
        events = []
        for e_low, e_high in zip(edges[0:], edges[1:]):
            e = self.expected_events(e_low, e_high, area, t_obs, solid_angle=solid_angle)
            events.append(e)

        events = np.array(events)
        return events


class CrabSpectrum(Spectrum):
    '''
    The gamma ray energy spectrum of the Crab Nebula as measured by the HEGRA experiment.
    See Aharonian, F. et al. (HEGRA collaboration) 2004, ApJ 614, 897
    '''

    def __init__(self, index=-2.62, normalization_constant=2.83e-14 / (u.GeV * u.cm**2 * u.s)):
        self.index = index
        self.normalization_constant = normalization_constant


class CosmicRaySpectrum(Spectrum):
    '''
    BESS Proton spectrum  ApJ 545, 1135 (2000) [arXiv:astro-ph/0002481],
    same as used by K. Bernloehr.

    This is the spectrum of cosmic protons.
    I stole this from the MARS Barcelona code provided by Tarek. H.
    '''

    def __init__(self, index=-2.7, normalization_constant=9.6e-9 / (u.GeV * u.cm**2 * u.s * u.sr)):
        self.index = index
        self.normalization_constant = normalization_constant


class CTAElectronSpectrum(Spectrum):
    '''
    See the IRF ASWG report page 22 and 23
    '''

    def __init__(self, index=-3.43, normalization_constant=2.385E-9 * u.Unit('cm-2 s-1 TeV-1 sr-1')):
        self.index = index
        self.normalization_constant = normalization_constant

    def flux(self, energy):

        energy = energy.to('TeV')
        N = self.normalization_constant * (energy / u.TeV)**(self.index)

        mu = -0.101
        sigma = 0.741
        f = 1.95
        b = (1 + f * (np.exp(norm.pdf(np.log10(energy / u.TeV), loc=mu, scale=sigma)) - 1))
        flux = N * b

        return flux.to(1 / (u.TeV * u.s * u.cm**2 * u.sr))

    def _integral(self, e_min, e_max):
        a = e_min.to(u.TeV).value
        b = e_max.to(u.TeV).value

        result, _ = quad(lambda e: self.flux(e * u.TeV).value, a, b)

        return result * self.normalization_constant.unit * u.TeV


class CosmicRaySpectrumPDG(Spectrum):
    '''
    PDG Cosmic Ray spectrum. This contains all nucleus types. (proton, helium, iron, ...)
    '''

    def __init__(self, index=-2.7, normalization_constant=1.8E4 / (0.001**(-2.7)) / (u.sr * u.s * u.m**2 * u.GeV)):
        self.index = index
        self.normalization_constant = normalization_constant



class CTAProtonSpectrum(Spectrum):
    '''
    Protons Spectrum as used by the CTA ASWG.
    '''

    def __init__(self, index=-2.62, normalization_constant=9.8E-6 / (u.sr * u.s * u.cm**2 * u.TeV)):
        self.index = index
        self.normalization_constant = normalization_constant


class MCSpectrum(Spectrum):
    '''
    A generic spectrum following a power law which can be used to get
    the number of simulated events generated by a Monte Carlo program
    or to reweight simulated events to another generic spectrum.

    Attributes
    ----------
        e_min:  Quantity
            Minimun energy simulated

        e_max: Quantity
            Maximum energy simulated

        total_showers_simulated: int
            Total number of showers that have been simulated

        generation_area: Quantity
            The total area over which the primary particles are scattered.
            Also know as the maximum_impact_distance**2 * pi.

        generator_solid_angle: Quantity (steradian)
            The solid angle over which the particles were created.
            This is necessary for extended sources like the cosmic ray spectrum

        generator_opening_angle: Quantity (deg or rad)
            The opening angle of the conde over which the particles were created.

        index: float
            The index with which the spectrum was generated. (e.g. -2.0)
    '''

    index = -2.0  # default for cta
    generator_solid_angle = 0 * u.sr
    generator_opening_angle = 0 * u.deg
    normalization_constant = 1 / (u.TeV * u.m**2 * u.s)

    @u.quantity_input(e_min=u.TeV, e_max=u.TeV, generation_area=u.m**2)
    def __init__(
            self,
            e_min,
            e_max,
            total_showers_simulated,
            generation_area,
            generator_opening_angle=None,
            index=-2.0  # default for CTA prod3
    ):
        '''
        To calculate the normalization constant of this spectrum some
        information about the event generator has to be specified.

        Parameters
        ----------
        e_min:  Quantity
            Minimun energy simulated
        e_max: Quantity
            Maximum energy simulated
        total_showers_simulated: int
            Total number of showers that have been simulated
        generation_area: Quantity
            The total area over which the primary particles are scattered.
            Also know as the maximum_impact_distance**2 * pi.
        generator_opening_angle: Quantity (deg or rad)
            The opening angle of the cone over which the particles were created.
            This is necessary for extended sources like the cosmic ray spectrum
        '''
        self.e_min = e_min.to('TeV')
        self.e_max = e_max.to('TeV')
        self.total_showers_simulated = total_showers_simulated
        self.index = index
        self.generation_area = generation_area.to('m^2')
        self.generator_opening_angle = generator_opening_angle
        self.normalization_constant = 1 / (u.TeV * u.m**2 * u.s)
        if generator_opening_angle is not None and generator_opening_angle > 0 * u.deg:
            self.normalization_constant = 1 / (u.TeV * u.m**2 * u.s * u.sr)
            angle = generator_opening_angle.to_value('rad')
            solid_angle = (1 - np.cos(angle)) * 2 * np.pi * u.sr
            self.generator_solid_angle = solid_angle

            N = self._integral(e_min.to('TeV'), e_max.to('TeV')) * (generation_area.to(u.m**2) * u.s * solid_angle)
            self.normalization_constant = (total_showers_simulated / N) / (u.TeV * u.m**2 * u.s * u.sr)

        else:
            N = self._integral(e_min.to('TeV'), e_max.to('TeV')) * (generation_area.to(u.m**2) * u.s)
            self.normalization_constant = (total_showers_simulated / N) / (u.TeV * u.m**2 * u.s)


    def __repr__(self):
        return f'E_Min:{self.e_min}, E_Max:{self.e_max}, N_total:{self.total_showers_simulated}, index:{self.index}, normalization contant:{self.normalization_constant}'

    def draw_energy_distribution(self, size):
        return super().draw_energy_distribution(
            e_min=self.e_min,
            e_max=self.e_max,
            size=size,
            index=self.index,
        )


    def expected_events_for_bins(self, energy_bins):
        '''
        Get the number of events which are expected to arrive from this spectral source.
        For each of the requested bins.
        Parameters
        ----------
        energy_bins: array like energy Quantity
            The energy binning to use.
        '''

        edges = energy_bins
        events = [self.expected_events(l, h) for (l, h) in zip(edges[0:], edges[1:])]
        events = np.array(events)
        return events


    def expected_events(self, e_min=None, e_max=None):
        if e_min is None:
            e_min = self.e_min
        if e_max is None:
            e_max = self.e_max
        return super().expected_events(
            e_min=e_min,
            e_max=e_max,
            area=self.generation_area,
            solid_angle=self.generator_solid_angle,
            t_obs=1 * u.s,
        )


    @classmethod
    def from_cta_runs(cls, runs):
        '''
        Get the spectrum object for the runs of a MC production.

        TODO: For now this assumes all runs have the been simulated with the same settings.

        Parameters
        ----------
        runs:  pandas.DataFrame
            table containing information about the runs for this MC production.
        '''
        mc_num_showers = runs.mc_num_showers.sum()
        # assume these numbers are equal for each run
        mc_spectral_index = runs.mc_spectral_index.iloc[0]
        mc_num_reuse = runs.mc_shower_reuse.iloc[0]
        mc_min_energy = runs.mc_energy_range_min.iloc[0] * u.TeV
        mc_max_energy = runs.mc_energy_range_max.iloc[0] * u.TeV
        generation_area = (runs.mc_max_scatter_range.iloc[0] * u.m)**2 * np.pi
        generator_opening_angle = (runs.mc_max_viewcone_radius.iloc[0] - runs.mc_min_viewcone_radius.iloc[0]) * u.deg

        return MCSpectrum(
            mc_min_energy,
            mc_max_energy,
            mc_num_showers * mc_num_reuse,
            generation_area,
            generator_opening_angle=generator_opening_angle,
            index=mc_spectral_index,
        )




    @u.quantity_input(event_energies=u.TeV, t_assumed_obs=u.h,)
    def reweigh_to_other_spectrum(
            self,
            other_spectrum,
            event_energies,
            t_assumed_obs,
    ):
        '''
        This method returns weights for the given events based on the given spectrum.
        '''

        if self.extended_source != other_spectrum.extended_source:
            raise ValueError('Both spectra must either be extended sources or not. No mixing. ')

        event_energies = event_energies.to('TeV')
        N = self.total_showers_simulated
        A = self.expected_events() * t_assumed_obs.to('s').value / N

        w = A * other_spectrum.flux(event_energies) / self.flux(event_energies)

        # at this point the value should have no units left
        assert w.si.unit.is_unity() is True
        return w.value

    def equivalent_obstime(self, other_spectrum):   
        n_events = self.total_showers_simulated
        integral_flux = other_spectrum._integral(self.e_min, self.e_max)
        area = self.generation_area
        t =  n_events / (area * integral_flux)
        if self.generator_solid_angle and self.generator_solid_angle.to_value(u.deg) > 0:
            solid_angle = self.generator_solid_angle
            solid_angle = (1 - np.cos(solid_angle)) * 2 * np.pi * u.sr
            t /= solid_angle
        return t.to(u.s)



if __name__ == '__main__':

    def trigger_efficency(simulated_energies, energy_bins):
        from scipy import interpolate
        p = [1.71e11, 0.0891, 1e5]

        xx = energy_bins.to('MeV').value
        efficency = p[0] * xx ** (-p[1]) * np.exp(-p[2] / xx)

        efficency = efficency / efficency.max()
        f = interpolate.interp1d(xx, efficency)
        r = np.random.uniform(0, 1, simulated_energies.shape)

        m = r > (1 - 0.5 * f(simulated_energies.to('MeV').value))
        print(f'total trigger efficiency: {m.sum() / len(simulated_energies)}')
        return m    



    crab = CrabSpectrum()

    e_min = 0.003 * u.TeV
    e_max = 300 * u.TeV
    area = 1 * u.km**2
    N = int(1E6)
    simulation_index = -2.0

    mc = MCSpectrum(
        e_min=e_min,
        e_max=e_max,
        total_showers_simulated=N,
        generation_area=area,
        index=simulation_index,
    )

    mc2 = MCSpectrum(
        e_min=e_min,
        e_max=e_max,
        total_showers_simulated=N,
        generation_area=area * 2,
        index=simulation_index,
    )
    print(mc.expected_events(), N)
    # assert mc2.expected_events() == N
    assert mc.generation_area * 2 ==  mc2.generation_area
    assert mc.expected_events() ==  mc2.expected_events()
    assert mc.normalization_constant == mc2.normalization_constant * 2
    assert mc.equivalent_obstime(crab) == mc2.equivalent_obstime(crab) * 2
    # executing this will create a plot which is usefull for checking if
    # the reweighing works correctly
    import matplotlib.pyplot as plt

    t_assumed_obs = 50 * u.h

    energy_bins, bin_center, bin_width = make_energy_bins(e_min=e_min, e_max=e_max, bins=20)

    mc = MCSpectrum(
        e_min=e_min,
        e_max=e_max,
        total_showers_simulated=N,
        generation_area=area,
        index=simulation_index,
    )

    random_energies = mc.draw_energy_distribution(N)


    events = crab.expected_events_for_bins(
        area=area,
        t_obs=t_assumed_obs,
        energy_bins=energy_bins
    )

    fig, [ax1, ax2] = plt.subplots(2, 1)

    ax1.errorbar(
        bin_center.value,
        events,
        xerr=bin_width.value * 0.5,
        linestyle='',
        marker='.',
        label='expected events from crab',
        color='black',
    )
    h, _, _ = ax1.hist(
        random_energies,
        bins=energy_bins,
        histtype='step',
        label='randomply sampled events with index {}'.format(simulation_index),
        color='gray',
    )
    w = mc.reweigh_to_other_spectrum(crab, random_energies, t_assumed_obs=t_assumed_obs)
    h_w, _, _ = ax1.hist(
        random_energies,
        bins=energy_bins,
        histtype='step',
        weights=w,
        label='reweighted energies',
        color='red',
        lw=2
    )




    trigger = trigger_efficency(random_energies, energy_bins)
    h_trigger, _, _ = ax1.hist(
        random_energies[trigger],
        bins=energy_bins,
        histtype='step',
        label='events seen by telescope',
        color='black',
        alpha=0.8,
    )
    h_trigger_w, _, _ = ax1.hist(
        random_energies[trigger],
        bins=energy_bins,
        weights=w[trigger],
        histtype='step',
        label='reiwghted events seen by telescope',
        color='black',
        lw=2,
    )

    plt.title('Event Reweighing')
    plt.suptitle('Red line should be on black points')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Energy in TeV')


    ax2.plot(bin_center, h / h_trigger)
    ax2.plot(bin_center, h_w / h_trigger_w)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel('ratios')
    ax2.set_xlabel('Energy in TeV')

    plt.show()
