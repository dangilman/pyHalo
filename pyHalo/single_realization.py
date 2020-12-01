from pyHalo.Halos.halo import Halo
from pyHalo.defaults import *
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.lensing_mass_function import LensingMassFunction
from pyHalo.Rendering.Main.SHMF_normalizations import *

from copy import deepcopy

def realization_at_z(realization, z, angular_coordinate_x=None, angular_coordinate_y=None, max_range=None):
    """

    :param realization: an instance of Realization
    :param z: the redshift where we want to extract halos
    :param angular_coordinate_x: if max_range is specified, will only keep halos within
    max_range of (angular_coordinate_x, angular_coordinate_y)
    :param angular_coordinate_y:
    :param max_range: radius in arcseconds where we want to keep halos. If None, will return a new realization class
     that contains all halos at redshift z the input realization class
    :return: a new instance of Realization
    """

    _halos, _indexes = realization.halos_at_z(z)
    halos = []
    indexes = []

    if max_range is not None:
        for i, halo in enumerate(_halos):
            dx, dy = halo.x - angular_coordinate_x, halo.y - angular_coordinate_y
            dr = (dx ** 2 + dy ** 2) ** 0.5
            if dr < max_range:
                halos.append(halo)
                indexes.append(i)
    else:
        halos = _halos
        indexes = _indexes

    return Realization.from_halos(halos, realization.halo_mass_function,
                                  realization._prof_params, realization._mass_sheet_correction,
                                  realization.rendering_classes), indexes

class Realization(object):

    def __init__(self, masses, x, y, r2d, r3d, mdefs, z, subhalo_flag, halo_mass_function,
                 halos=None, other_params={}, mass_sheet_correction=True, dynamic=False,
                 rendering_classes=None):

        """

        This class is the main class that stores information regarding realizations of dark matter halos. It is not
        intended to be created directly by the user. Instances of this class are created through the class
        pyHalo/pyHalo_dynamic.

        :param masses: an array of halo masses (units solar mass)
        :param x: an array of halo x-coordinates (units arcsec)
        :param y: an array of halo y-coordinates (units arcsec)
        :param r2d: an array of halo 2-d distances from lens center (units kpc / (kpc/arsec), or arcsec,
        at halo redshift)
        :param r3d: an array of halo 2-d distances from lens center (units kpc / (kpc/arsec), or arcsec,
        at halo redshift)
        :param mdefs: mass definition of each halo
        :param z: halo redshift
        :param subhalo_flag: whether each halo is a subhalo or a regular halo
        :param halo_mass_function: an instance of LensingMassFunction (see Cosmology.LensingMassFunction)
        :param halos: a list of halo class instances
        :param other_params: kwargs for the realiztion
        :param mass_sheet_correction: whether to apply a mass sheet correction
        :param dynamic: whether the realization is rendered with pyhalo_dynamic or not
        :param rendering_classes: a list of rendering class instances
        """

        self._mass_sheet_correction = mass_sheet_correction

        self.halo_mass_function = halo_mass_function
        self.geometry = halo_mass_function.geometry
        self.lens_cosmo = LensCosmo(self.geometry._zlens, self.geometry._zsource,
                                    self.geometry._cosmo)

        self._lensing_functions = []
        self.halos = []
        self._loaded_models = {}
        self._has_been_shifted = False

        self._prof_params = set_default_kwargs(other_params, dynamic, self.geometry._zsource)

        if halos is None:

            for mi, xi, yi, r2di, r3di, mdefi, zi, sub_flag in zip(masses, x, y, r2d, r3d,
                           mdefs, z, subhalo_flag):

                self._add_halo(mi, xi, yi, r2di, r3di, mdefi, zi, sub_flag)

            if self._prof_params['subhalos_of_field_halos']:
                raise Exception('subhalos of halos not yet implemented.')

        else:

            for halo in halos:
                self._add_halo(None, None, None, None, None, None, None, None, halo=halo)

        self._reset()

        self.set_rendering_classes(rendering_classes)

    @classmethod
    def from_halos(cls, halos, halo_mass_function, prof_params, msheet_correction, rendering_classes):

        """

        :param halos: a list of halo class instances
        :param halo_mass_function: an instance of LensingMassFunction (see Cosmology.LensingMassFunction)
        :param prof_params: keyword arguments for the realization
        :param msheet_correction: whether or not to apply a mass sheet correction
        :param rendering_classes: a list of rendering classes
        :return: an instance of Realization created directly from the halo class instances
        """

        realization = Realization(None, None, None, None, None, None, None, None, halo_mass_function,
                                  halos=halos, other_params=prof_params,
                                  mass_sheet_correction=msheet_correction,
                                  rendering_classes=rendering_classes)

        return realization

    def set_rendering_classes(self, rendering_classes):

        self.rendering_classes = rendering_classes

    def join(self, real):
        """

        :param real: another realization, possibly a filtered version of self
        :return: a new realization with all unique halos from self and real
        """
        halos = []

        tags = self._tags(self.halos)
        real_tags = self._tags(real.halos)
        if len(tags) >= len(real_tags):
            long, short = tags, real_tags
            halos_long, halos_short = self.halos, real.halos
        else:
            long, short = real_tags, tags
            halos_long, halos_short = real.halos, self.halos

        for halo in halos_short:
            halos.append(halo)

        for i, tag in enumerate(long):

            if tag not in short:
                halos.append(halos_long[i])

        return Realization.from_halos(halos, self.halo_mass_function, self._prof_params,
                                      self._mass_sheet_correction, self.rendering_classes)

    def _tags(self, halos=None):

        if halos is None:
            halos = self.halos
        tags = []

        for halo in halos:

            tags.append(halo._unique_tag)

        return tags

    def _reset(self):

        self.x = []
        self.y = []
        self.masses = []
        self.redshifts = []
        self.r2d = []
        self.r3d = []
        self.mdefs = []
        self._halo_tags = []
        self.subhalo_flags = []

        for halo in self.halos:
            self.masses.append(halo.mass)
            self.x.append(halo.x)
            self.y.append(halo.y)
            self.redshifts.append(halo.z)
            self.r2d.append(halo.r2d)
            self.r3d.append(halo.r3d)
            self.mdefs.append(halo.mdef)
            self._halo_tags.append(halo._unique_tag)
            self.subhalo_flags.append(halo.is_subhalo)

        self.masses = np.array(self.masses)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.r2d = np.array(self.r2d)
        self.r3d = np.array(self.r3d)
        self.redshifts = np.array(self.redshifts)

        self.unique_redshifts = np.unique(self.redshifts)

    def shift_background_to_source(self, ray_interp_x, ray_interp_y):

        """

        :param ray_interp_x: instance of scipy.interp1d, returns the angular position of a ray
        fired through the lens center given a comoving distance
        :param ray_interp_y: same but for the y coordinate
        :return:
        """

        # add all halos in front of main deflector with positions unchanged
        halos = []

        if self._has_been_shifted:
            return self

        for halo in self.halos:

            comoving_distance_z = self.lens_cosmo.cosmo.D_C_z(halo.z)

            xshift, yshift = ray_interp_x(comoving_distance_z), ray_interp_y(comoving_distance_z)

            new_x, new_y = halo.x + xshift, halo.y + yshift

            new_halo = Halo(mass=halo.mass, x=new_x, y=new_y, r2d=halo.r2d, r3d=halo.r3d, mdef=halo.mdef, z=halo.z,
                        sub_flag=halo.is_subhalo, cosmo_m_prof=self.lens_cosmo,
                        args=self._prof_params)
            halos.append(new_halo)

        new_realization = Realization.from_halos(halos, self.halo_mass_function, self._prof_params,
                                      self._mass_sheet_correction,
                                      rendering_classes=self.rendering_classes)

        new_realization._has_been_shifted = True

        return new_realization

    def _add_halo(self, m, x, y, r2, r3, md, z, sub_flag, halo=None):
        if halo is None:

            halo = Halo(mass=m, x=x, y=y, r2d=r2, r3d=r3, mdef=md, z=z, sub_flag=sub_flag, cosmo_m_prof=self.lens_cosmo,
                        args=self._prof_params)
        self._lensing_functions.append(self._lens(halo))
        self.halos.append(halo)

    def lensing_quantities(self, return_kwargs=False, add_mass_sheet_correction=True):

        kwargs_lens = []
        lens_model_names = []
        redshift_list = []
        kwargs_lensmodel = None

        for i, halo in enumerate(self.halos):

            args = tuple([halo.x, halo.y, halo.mass, halo.z] + halo.profile_args)

            kw, model_args = self._lensing_functions[i].params(*args)

            lenstronomy_ID = self._lensing_functions[i].lenstronomy_ID

            lens_model_names.append(lenstronomy_ID)
            kwargs_lens.append(kw)
            redshift_list += [halo.z]

            if kwargs_lensmodel is None:
                kwargs_lensmodel = model_args
            else:

                if model_args is not None and not (type(model_args) is type(kwargs_lensmodel)):
                    raise Exception('Currently only one numerical lens class at once is supported.')

        if self._mass_sheet_correction and add_mass_sheet_correction:

            if self.rendering_classes is None:
                raise Exception('if applying a convergence sheet correction, must specify '
                                'the rendering classes.')

            kwargs_mass_sheets, profile_list, z_sheets = self.mass_sheet_correction(self.rendering_classes)
            kwargs_lens += kwargs_mass_sheets
            lens_model_names += profile_list
            redshift_list = np.append(redshift_list, z_sheets)

        if return_kwargs:
            return {'lens_model_list': lens_model_names,
                    'lens_redshift_list': redshift_list,
                    'z_source': self.geometry._zsource,
                    'z_lens': self.geometry._zlens,
                    'multi_plane': True}, kwargs_lens
        else:
            return lens_model_names, redshift_list, kwargs_lens, kwargs_lensmodel

    def mass_sheet_correction(self, rendering_classes):

        kwargs_mass_sheets = []

        redshifts = []

        profiles = []

        if self._prof_params['subtract_exact_mass_sheets']:

            kwargs_mass_sheets = [{'kappa_ext': -self.mass_at_z_exact(zi) / self.lens_cosmo.sigma_crit_mass(zi, self.geometry)}
                                     for zi in self.unique_redshifts]

            redshifts = self.unique_redshifts

            profiles = ['CONVERGENCE'] * len(kwargs_mass_sheets)

        else:

            for rendering_class in rendering_classes:

                kwargs_new, profiles_new, redshifts_new = \
                    rendering_class.negative_kappa_sheets_theory()

                kwargs_mass_sheets += kwargs_new
                redshifts += redshifts_new
                profiles += profiles_new

        return kwargs_mass_sheets, profiles, redshifts

    def _lens(self, halo):

        if halo.mdef not in self._loaded_models.keys():

            model = self._load_model(halo)
            self._loaded_models.update({halo.mdef: model})

        return self._loaded_models[halo.mdef]

    def _load_model(self, halo):

        if halo.mdef == 'NFW':
            from pyHalo.Lensing.NFW import NFWLensing
            lens = NFWLensing(self.lens_cosmo)

        elif halo.mdef == 'TNFW':
            from pyHalo.Lensing.NFW import TNFWLensing
            lens = TNFWLensing(self.lens_cosmo)

        elif halo.mdef == 'SIDM_TNFW':
            from pyHalo.Lensing.coredTNFW import coreTNFW
            lens = coreTNFW(self.lens_cosmo)

        elif halo.mdef == 'PT_MASS':
            from pyHalo.Lensing.PTmass import PTmassLensing
            lens = PTmassLensing(self.lens_cosmo)

        elif halo.mdef == 'SIS':
            from pyHalo.Lensing.sis import SISLensing
            lens = SISLensing(self.lens_cosmo)

        elif halo.mdef == 'PJAFFE':

            from pyHalo.Lensing.pjaffe import PJaffeLensing
            lens = PJaffeLensing(self.lens_cosmo)

        else:
            raise ValueError('halo profile ' + str(halo.mdef) + ' not recongnized.')

        return lens

    def halo_physical_coordinates(self, halos):

        xcoords, ycoords, masses, redshifts = [], [], [], []

        for halo in halos:
            D = self.lens_cosmo.cosmo.D_C_transverse(halo.z)
            x_arcsec, y_arcsec = halo.x, halo.y
            x_comoving, y_comoving = D * x_arcsec, D * y_arcsec
            xcoords.append(x_comoving)
            ycoords.append(y_comoving)
            masses.append(halo.mass)
            redshifts.append(halo.z)
        return np.array(xcoords), np.array(ycoords), np.log10(masses), np.array(redshifts)

    def add_halo(self, mass, x, y, r2d, r3d, mdef, z, sub_flag):

        new_real = Realization([mass], [x], [y], [r2d], [r3d], [mdef], [z], [sub_flag], self.halo_mass_function,
                               halos = None, other_params=self._prof_params,
                               mass_sheet_correction=self._mass_sheet_correction)

        realization = self.join(new_real)
        return realization

    def add_halos(self, masses, x, y, r2d, r3d, mdefs, z, sub_flags):

        """
        Added this routine to maintin backwards compatability

        """
        new_real = Realization(masses, x, y, r2d, r3d, mdefs, z, sub_flags, self.halo_mass_function,
                               halos=None, other_params=self._prof_params,
                               mass_sheet_correction=self._mass_sheet_correction)

        realization = self.join(new_real)
        return realization

    def change_profile_params(self, new_args):

        new_params = deepcopy(self._prof_params)
        new_params.update(new_args)

        return Realization(self.masses, self.x, self.y, self.r2d, self.r3d, self.mdefs,
                           self.redshifts, self.subhalo_flags, self.halo_mass_function,
                           other_params=new_params, mass_sheet_correction=self._mass_sheet_correction)

    def change_mdef(self, new_mdef):

        new_halos = []
        for halo in self.halos:
            duplicate = deepcopy(halo)
            if duplicate.mdef == 'cNFWmod_trunc' and new_mdef == 'TNFW':

                duplicate._mass_def_arg = duplicate.profile_args[0:-1]

            else:
                raise Exception('combination '+duplicate.mdef + ' and '+
                                    new_mdef+' not recognized.')

            duplicate.mdef = new_mdef
            new_halos.append(duplicate)

        return Realization(None, None, None, None, None, None, None, None, self.halo_mass_function,
                           halos = new_halos, other_params= self._prof_params,
                           mass_sheet_correction=self._mass_sheet_correction)

    def split_at_z(self, z):

        halos_1, halos_2 = [], []
        for halo in self.halos:
            if halo.z <= z:
                halos_1.append(halo)
            else:
                halos_2.append(halo)

        realization_1 = Realization.from_halos(halos_1, self.halo_mass_function,
                                               self._prof_params, self._mass_sheet_correction, self.rendering_classes)
        realization_2 = Realization.from_halos(halos_2, self.halo_mass_function,
                                               self._prof_params, self._mass_sheet_correction, self.rendering_classes)

        return realization_1, realization_2

    def filter_by_mass(self, mlow):

        halos = []
        for halo in self.halos:
            if halo.mass >= mlow:
                halos.append(halo)

        return Realization.from_halos(halos, self.halo_mass_function,
                                      self._prof_params, self._mass_sheet_correction, self.rendering_classes)

    def filter(self, aperture_radius_front,
                   aperture_radius_back,
                   mass_allowed_in_apperture_front,
                   mass_allowed_in_apperture_back,
                   mass_allowed_global_front,
                   mass_allowed_global_back,
                   interpolated_x_angle, interpolated_y_angle,
                    zmin=None, zmax=None):

        halos = []

        if zmax is None:
            zmax = self.geometry._zsource
        if zmin is None:
            zmin = 0

        for plane_index, zi in enumerate(self.unique_redshifts):

            plane_halos, _ = self.halos_at_z(zi)
            inds_at_z = np.where(self.redshifts == zi)[0]
            x_at_z = self.x[inds_at_z]
            y_at_z = self.y[inds_at_z]
            masses_at_z = self.masses[inds_at_z]

            if zi < zmin:
                continue
            if zi > zmax:
                continue

            comoving_distance_z = self.lens_cosmo.cosmo.D_C_z(zi)

            if zi <= self.geometry._zlens:

                minimum_mass_everywhere = deepcopy(mass_allowed_global_front)
                minimum_mass_in_window = deepcopy(mass_allowed_in_apperture_front)
                position_cut_in_window = deepcopy(aperture_radius_front)

            else:

                minimum_mass_everywhere = deepcopy(mass_allowed_global_back)
                minimum_mass_in_window = deepcopy(mass_allowed_in_apperture_back)
                position_cut_in_window = deepcopy(aperture_radius_back)

            keep_inds_mass = np.where(masses_at_z >= 10 ** minimum_mass_everywhere)[0]

            inds_m_low = np.where(masses_at_z < 10 ** minimum_mass_everywhere)[0]

            keep_inds_dr = []
            for idx in inds_m_low:
                for k, (interp_x, interp_y) in enumerate(zip(interpolated_x_angle, interpolated_y_angle)):

                    dx = x_at_z[idx] - interp_x(comoving_distance_z)
                    dy = y_at_z[idx] - interp_y(comoving_distance_z)
                    dr = np.sqrt(dx ** 2 + dy ** 2)
                    if dr <= position_cut_in_window:
                        keep_inds_dr.append(idx)
                        break

            keep_inds = np.append(keep_inds_mass, np.array(keep_inds_dr)).astype(int)

            tempmasses = masses_at_z[keep_inds]
            keep_inds = keep_inds[np.where(tempmasses >= 10 ** minimum_mass_in_window)[0]]

            for halo_index in keep_inds:
                halos.append(plane_halos[halo_index])

        return Realization.from_halos(halos, self.halo_mass_function, self._prof_params,
                                      self._mass_sheet_correction, self.rendering_classes)

    def halos_at_z(self,z):

        halos = []
        index = []
        for i, halo in enumerate(self.halos):
            if halo.z != z:
                continue
            halos.append(halo)
            index.append(i)

        return halos, index

    def mass_at_z_exact(self, z):

        inds = np.where(self.redshifts == z)
        m_exact = np.sum(self.masses[inds])
        return m_exact

    def number_of_halos_before_redshift(self, z):
        n = 0
        for halo in self.halos:
            if halo.z < z:
                n += 1
        return n

    def number_of_halos_after_redshift(self, z):
        n = 0
        for halo in self.halos:
            if halo.z > z:
                n += 1
        return n

    def number_of_halos_at_redshift(self, z):
        n = 0
        for halo in self.halos:
            if halo.z == z:
                n += 1
        return n

class SingleHalo(Realization):

    """
    A quick and dirty class useful for generating a realization with a few
    user-specified halos.
    """

    def __init__(self, halo_mass, x, y, r3d, mdef, z, zlens, zsource, subhalo_flag=True,
                 cone_opening_angle=6, log_mlow=6, log_mhigh=10,
                 kwargs_halo={}, cosmo=None):

        r2d = np.sqrt(x ** 2 + y ** 2)

        if cosmo is None:
            cosmo = Cosmology()
        halo_mass_function = LensingMassFunction(cosmo, 10**log_mlow, 10**log_mhigh,
                                                 zlens, zsource, cone_opening_angle, use_lookup_table=True)

        kwargs_halo.update({'cone_opening_angle': cone_opening_angle})
        super(SingleHalo, self).__init__([halo_mass], [x], [y], [r2d],
                                         [r3d], [mdef], [z], [subhalo_flag], halo_mass_function,
                                         other_params=kwargs_halo, mass_sheet_correction=False)

def add_core_collapsed_subhalos(f_collapsed, realization):

    halos = realization.halos

    for index, halo in enumerate(halos):
        if halo.is_subhalo:
            u = np.random.rand()
            if u < f_collapsed:
                # change mass definition
                new_halo = Halo.change_profile_definition(halo, 'PJAFFE')
                halos[index] = new_halo

    halo_mass_function = realization.halo_mass_function
    prof_params = realization._prof_params
    msheet_correction = realization._mass_sheet_correction
    rendering_classes = realization.rendering_classes

    return Realization.from_halos(halos, halo_mass_function, prof_params,
                                  msheet_correction, rendering_classes)


