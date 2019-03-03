import matplotlib.pyplot as plt
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D, Background2D
from gammapy.irf import PSF3D

fig, [top, center, bottom] = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)

irf_filename = 'cta_irf.fits'


aeff = EffectiveAreaTable2D.read(irf_filename, hdu="EFFECTIVE AREA")
aeff.plot(ax=top[0], add_cbar=True)
aeff.plot_energy_dependence(ax=top[1])
aeff.plot_offset_dependence(ax=top[2])


bkg = Background2D.read(irf_filename)
bkg.plot(ax=center[0])
print(bkg)

psf = PSF3D.read(irf_filename, hdu='PSF')
# psf.plot_psf_vs_rad(ax=center[0])

table = psf.to_table_psf(energy='1 TeV', theta='0 deg')
table.plot_psf_vs_rad(ax=center[1])
# psf.plot_containment(ax=center[0])  
# psf.plot_containment_vs_energy(ax=center[2])  

edisp = EnergyDispersion2D.read(irf_filename, hdu="ENERGY DISPERSION")
edisp.plot_bias(ax=bottom[0])
edisp.plot_migration(ax=bottom[1])

plt.show()