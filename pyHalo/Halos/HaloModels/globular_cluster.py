from pyHalo.Halos.halo_base import Halo
from lenstronomy.LensModel.Profiles.king import King

"""
=====================================================
Notes on the observed GC properties behind the default parameters and the
King (1962) profile. Values are for early-type galaxies (ETGs, the relevant
lens hosts). Derived (non-directly-measured) quantities are flagged.

1. SIZE  (gc_size_mean, gc_size_sigma, size-mass scaling) in RealizationExtensions
   - Median half-light radius r_h ~ 2.7 +/- 0.35 pc in ETGs; consistent with
     the Milky Way median (~3 pc). Basis for gc_size_mean = 3.0 pc.
       Jordan et al. 2005, ApJ 634, 1002 (ACSVCS X); Masters et al. 2010,
       ApJ 715, 1419 (ACS Fornax VII).
   - r_h shows NO significant trend with cluster mass over the classical GC
     regime. Sizes only rise above the UCD transition at
     M ~ 2e6 Msun (star-cluster/UCD boundary, plausibly set by a maximum
     stellar surface density).
       Misgeld & Hilker 2011, MNRAS 414, 3699; Norris et al. 2014, MNRAS 443, 1151.
   - Scatter ~0.2-0.3 dex at fixed mass; applied as log-normal (gc_size_sigma
     ~ 0.2 dex). Because peak kappa ~ Sigma_0 ~ M / r_h^2 (Jordan et al. 2005)

2. MASS FUNCTION  (log10_mgc_mean, log10_mgc_sigma)
   - GCLF turnover mass M_TO ~ (2.2 +/- 0.4)e5 Msun (log10 ~ 5.34) in bright
     galaxies; assume M/L_V ~ 1.5-2 -> log10_mgc_mean ~ 5.3.
   - Dispersion grows with host luminosity: sigma ~ 1.2-1.4 mag (~0.5-0.55 dex)
     in giant Es (MW's ~0.4 dex is too narrow) -> log10_mgc_sigma ~ 0.5-0.55.
   - Log-normal is fine near the turnover; the bright end (>~1e6.5 Msun tail:
     omega Cen analogs, stripped nuclei, UCDs) is better fit by an evolved
     Schechter function.
       Jordan et al. 2007, ApJS 171, 101 (ACSVCS XII).

3. TOTAL MASS / SURFACE DENSITY  (gc_surface_mass_density)
   - GC-system-to-halo-mass ratio eta = M_GCS/M_halo ~ (3-4)e-5, ~constant
     over a wide halo-mass range. For M_halo ~ 2e13 Msun -> M_GCS ~ 6-8e8 Msun.
       Hudson, Harris & Harris 2014, ApJL 787, L5; Harris, Blakeslee & Harris
       2017, ApJ 836, 67.
   - GC systems follow Sersic/de Vaucouleurs profiles, 2-4x more extended than
     the host light; R_e,GCS scales ~ R_vir (~ M_halo^1/3), ~20 kpc for such a
     lens. Red (metal-rich) subpop is more centrally concentrated than blue; at
     5-10 kpc one samples mostly red.
       Forbes 2017, MNRASL 472, L104.
   - DERIVED: projecting M_GCS ~ 6-8e8 with R_e,GCS ~ 20 kpc and Sigma ~ R^-1.5
     gives Sigma_GC(5-10 kpc) ~ 10^5.4-10^5.7 Msun/kpc^2 -> default 10^5.6.
     Caveats: (i) Sigma_GC drops ~2-3x from 5->10 kpc, so scale per annulus;
     (ii) ~0.3 dex galaxy-to-galaxy scatter in eta, BCG/M87-class ~5x higher
     (10^6.2-6.5) -> treat as a prior, not a constant.

4. INTERNAL STRUCTURE  (gc_concentration_mean/sigma -> c; central density Sigma_0)
   - GCs are well described by King (1962) models with a finite tidal cutoff.
     This profile uses the King (1962) empirical surface-density form, which
     truncates naturally at the tidal radius r_t -- no steep outer power law is
     needed to stand in for the truncation.
       King 1962, AJ 67, 471; McLaughlin & van der Marel 2005, ApJS 161, 304.
   - The King concentration c = log10(r_t / r_c) sets the profile shape. MW GCs
     span c ~ 0.7-2.5 with a median ~1.5; core-collapsed clusters sit at the
     high end (formally c -> 2.5). Basis for gc_concentration_mean ~ 1.5
     (r_t/r_c ~ 30), applied as log-normal with ~0.1-0.3 dex scatter.
       Harris 1996, AJ 112, 1487 (2010 edition); McLaughlin & van der Marel 2005.
   - Size and concentration are INDEPENDENT parameters. The (well-measured)
     projected half-mass radius r_h fixes the core radius via r_c = r_h / f(c),
     where f(c) = R_h / r_c is the dimensionless projected half-mass radius in
     core units -- a function of c only, obtained by inverting the enclosed-mass
     profile. This decouples the observed size r_h from the structural
     concentration c (unlike a single-scale r_h ~ 0.77 r_c fix).
   - Central surface density Sigma_0 ~ 1e3-1e5 Msun/pc^2 (up to ~1e6 for
     core-collapsed systems). Here Sigma_0 is NOT a free input: it is set by the
     total (projected) mass together with (r_h, c) via sigma0_from_mass_2d,
     Sigma_0 = M_2D / [2 pi r_c^2 F(x_t) / (1 - a)^2],  x_t = r_t/r_c = 10^c,
     a = (1 + 10^(2c))^(-1/2).
       Harris 1996; McLaughlin & van der Marel 2005.

References
---------
King 1962, AJ 67, 471                     (empirical King profile)
Jordan et al. 2005, ApJ 634, 1002        (ACSVCS X, half-light radii)
Masters et al. 2010, ApJ 715, 1419       (ACS Fornax VII, half-light radii)
Misgeld & Hilker 2011, MNRAS 414, 3699   (star-cluster/UCD boundary)
Norris et al. 2014, MNRAS 443, 1151      (AIMSS I, mass-size relation)
Jordan et al. 2007, ApJS 171, 101        (ACSVCS XII, GCLF, evolved Schechter)
Hudson, Harris & Harris 2014, ApJL 787, L5   (GC system - halo mass)
Harris, Blakeslee & Harris 2017, ApJ 836, 67 (GC system - halo mass)
Forbes 2017, MNRASL 472, L104            (GC system sizes vs halo)
McLaughlin & van der Marel 2005, ApJS 161, 304 (structural parameters)
Harris 1996, AJ 112, 1487 (2010 edition) (MW GC catalog, King concentrations)
"""

_KING = King()

class GlobularClusterKing(Halo):

    def __init__(self, mass, x, y, z, lens_cosmo_instance, args, unique_tag):
        """
        A globular cluster modeled using the empirical (analytic) approximation of a King profile (King 1962)
        :param mass: total mass
        :param x: x coordinate [arcsec]
        :param y: y coordinate [arcsec]
        :param z: redshift
        :param lens_cosmo_instance: an instance of LensCosmo class
        :param args: keyword arguments for the profile, must contain 'r_h', and
            'c', the concentration log10(r_t / r_c) where r_c is the GC core size
        :param unique_tag: number associated with class
        """
        self._prof = _KING
        self._lens_cosmo = lens_cosmo_instance
        mdef = 'KING'
        is_subhalo = False
        super(GlobularClusterKing, self).__init__(mass, x, y, None, mdef, z, is_subhalo,
                                                 lens_cosmo_instance, args, unique_tag,
                                                 fixed_position=True)

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_lenstronomy_args'):
            kpc_per_arcsec = self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)
            sigma0_pc, r_h_pc, c = self.profile_args
            sigma_crit_mpc = self._lens_cosmo.get_sigma_crit_lensing(self.z, self._lens_cosmo.z_source)
            sigma_crit_pc = sigma_crit_mpc * 1e-12
            sigma0 = sigma0_pc / sigma_crit_pc
            r_h_arcsec = 1e-3 * r_h_pc / kpc_per_arcsec
            self._lenstronomy_args = [{'sigma0': sigma0,
                                       'r_h': r_h_arcsec,
                                       'center_x': self.x, 'center_y': self.y,
                                       'c': c}]
        return self._lenstronomy_args, None

    def density_profile_2d(self, r):
        """
        Computes the projected (2-D) mass density profile of the cluster with lenstronomy
        :param r: projected distance from center of the cluster [kpc]
        :return: the projected mass density in units M_sun / kpc^2
        """
        sigma0_pc, r_h_pc, c = self.profile_args
        sigma_pc = self._prof.density_2d(1e3 * r, 0.0, sigma0_pc, r_h_pc, c)
        return 1e6 * sigma_pc

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            r_h_pc = self._args['r_h']  # in parsec
            c = self._args['c']
            sigma0_pc = float(self._prof.sigma0_from_mass_2d(self.mass, r_h_pc, c))
            self._profile_args = (sigma0_pc, r_h_pc, c)
        return self._profile_args

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        return ['KING']

    @property
    def z_eval(self):
        """
        Returns the redshift at which to evalate the concentration-mass relation
        """
        return self.z
