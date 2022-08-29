from scipy.interpolate import interp1d
from pyHalo.defaults import *
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.HaloModels.NFW import NFWSubhhalo, NFWFieldHalo
from pyHalo.Halos.HaloModels.TNFW import TNFWFieldHalo, TNFWSubhalo
from pyHalo.Halos.HaloModels.generalized_nfw import GeneralNFWFieldHalo, GeneralNFWSubhalo
from pyHalo.Halos.HaloModels.powerlaw import PowerLawFieldHalo, PowerLawSubhalo
from pyHalo.Halos.HaloModels.PsuedoJaffe import PJaffeSubhalo
from pyHalo.Halos.HaloModels.PTMass import PTMass
from pyHalo.Halos.HaloModels.coreTNFW import coreTNFWFieldHalo, coreTNFWSubhalo
from pyHalo.Halos.HaloModels.coreNFW import coreNFWFieldHalo, coreNFWSubhalo
from pyHalo.Halos.HaloModels.ULDM import ULDMFieldHalo, ULDMSubhalo
from pyHalo.Halos.HaloModels.gaussian import Gaussian
import numpy as np
from copy import deepcopy


def realization_at_z(realization, z, angular_coordinate_x=None, angular_coordinate_y=None, max_range=None,
                     mass_sheet_correction=True):
    """
    :param realization: an instance of Realization
    :param z: the redshift where we want to extract halos
    :param angular_coordinate_x: if max_range is specified, will only keep halos within
    max_range of (angular_coordinate_x, angular_coordinate_y)
    :param angular_coordinate_y:
    :param max_range: radius in arcseconds where we want to keep halos. If None, will return a new realization class
     that contains all halos at redshift z contained in the input realization class
    :param mass_sheet_correction: whether or not to include convergence sheet correction in returned realization
    :return: a new instance of Realization, and the indexes of halos that were kept from the original realization
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
                indexes.append(_indexes[i])
    else:
        halos = _halos
        indexes = _indexes

    centerx, centery = realization.rendering_center

    return Realization.from_halos(halos, realization.lens_cosmo,
                                  realization._prof_params,
                                  mass_sheet_correction,
                                  realization.rendering_classes,
                                  centerx, centery), indexes

class Realization(object):

    """
    This is the main class for storing a population of dark matter halos, both in the main lens plane and along the
    line of sight. This class is created by the main pyhalo module.
    """

    def __init__(self, masses, x, y, r3d, mdefs, z, subhalo_flag, lens_cosmo,
                 halos=None, kwargs_realization={}, mass_sheet_correction=True,
                 rendering_classes=None, rendering_center_x=None, rendering_center_y=None,
                 geometry=None):

        """

        This class is the main class that stores information regarding realizations of dark matter halos. It is not
        intended to be created directly by the user. Instances of this class are created through the class
        pyHalo or pyHalo_dynamic.

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
        :param lens_cosmo: an instance of LensCosmo (see Halos.lens_cosmo)
        :param halos: a list of halo class instances
        :param kwargs_realization: kwargs for the realiztion
        :param mass_sheet_correction: whether to apply a mass sheet correction
        :param rendering_classes: a list of rendering class instances
        :param rendering_center_x: an instance of scipy.interp1d that returns an angular position given a comoving distance.
        The angular coordinate defines the center of the rendering volume, and halos will be distributed symmetrically around it.
        The value defaults to 0, but is overridden when the method shift_background_to_source is called
        :param rendering_center_y: same as rendering_center_x, but for the y angular coordinate
        :param geometry: (optional, only relevant is subtract_exact_mass_sheets=True is specified in kwargs_realization)
        an instance of Geometry (pyHalo.Cosmology.geometry) that defines the rendering volume
        """

        self.apply_mass_sheet_correction = mass_sheet_correction
        self.geometry = geometry
        self.lens_cosmo = lens_cosmo
        self._zlens, self._zsource = self.lens_cosmo.z_lens, self.lens_cosmo.z_source
        self.astropy_instance = self.lens_cosmo.cosmo.astropy
        self.halos = []
        self._loaded_models = {}
        self._has_been_shifted = False
        self._prof_params = set_default_kwargs(kwargs_realization, self._zsource)

        if halos is None:

            for mi, xi, yi, r3di, mdefi, zi, sub_flag in zip(masses, x, y, r3d,
                           mdefs, z, subhalo_flag):

                unique_tag = np.random.rand()
                model = self._load_halo_model(mi, xi, yi, r3di, mdefi, zi, sub_flag, self.lens_cosmo,
                                              self._prof_params, unique_tag)
                self.halos.append(model)


        else:

            self.halos = halos

        self._reset()

        self.set_rendering_classes(rendering_classes)

        if rendering_center_x is None or rendering_center_y is None:
            _z = np.linspace(0, self._zsource, 100)
            d = [self.lens_cosmo.cosmo.D_C_transverse(zi) for zi in _z]
            angle = np.zeros_like(d)
            rendering_center_x = interp1d(d, angle)
            rendering_center_y = interp1d(d, angle)

        self._rendering_center_x = rendering_center_x
        self._rendering_center_y = rendering_center_y

    @classmethod
    def from_halos(cls, halos, lens_cosmo, prof_params, msheet_correction, rendering_classes,
                   rendering_center_x=None, rendering_center_y=None, geometry=None):

        """

        :param halos: a list of halo class instances
        :param lens_cosmo: an instance of LensCosmo (see Halos.lens_cosmo)
        :param prof_params: keyword arguments for the realization
        :param msheet_correction: whether or not to apply a mass sheet correction
        :param rendering_classes: a list of rendering classes
        :param rendering_center_x: an instance of scipy.interp1d that returns an angular position given a comoving distance.
        The angular coordinate defines the center of the rendering volume, and halos will be distributed symmetrically around it.
        The value defaults to 0, but is overridden when the method shift_background_to_source is called
        :param rendering_center_y: same as rendering_center_x, but for the y angular coordinate
        :param geometry: (optional, only relevant is subtract_exact_mass_sheets=True is specified in kwargs_realization)
        an instance of Geometry (pyHalo.Cosmology.geometry) that defines the rendering volume
        :return: an instance of Realization created directly from the halo class instances
        """

        realization = Realization(None, None, None, None, None, None, None, lens_cosmo,
                                  halos=halos, kwargs_realization=prof_params,
                                  mass_sheet_correction=msheet_correction,
                                  rendering_classes=rendering_classes,
                                  rendering_center_x=rendering_center_x,
                                  rendering_center_y=rendering_center_y,
                                  geometry=geometry)

        return realization

    @property
    def rendering_center(self):

        """
        Returns the instances of scipy.interp1d that compute the coordinate center of the lensing volume given a comoving
        distance.
        """
        return self._rendering_center_x, self._rendering_center_y

    def filter(self, aperture_radius_front,
               aperture_radius_back,
               log_mass_allowed_in_aperture_front,
               log_mass_allowed_in_aperture_back,
               log_mass_allowed_global_front,
               log_mass_allowed_global_back,
               interpolated_x_angle, interpolated_y_angle,
               zmin=None, zmax=None, aperture_units='ANGLES'):

        """

        :param aperture_radius_front: the radius of a circular window around each light ray where halos are halo kept
        if they are more massive than log_mass_allowed_in_aperture_front (applied for z < z_lens)
        :param aperture_radius_back: the radius of a circular window around each light ray where halos are halo kept
        if they are more massive than log_mass_allowed_in_aperture_back (applied for z < z_lens)
        :param log_mass_allowed_in_aperture_front: the minimum halo mass to be kept inside the tube around each light ray
        in the foreground
        :param log_mass_allowed_in_aperture_back: the minimum halo mass to be kept inside the tube around each light ray
        in the background
        :param log_mass_allowed_global_front: The minimum mass to be kept everywhere in the foreground (if this is smaller
        than log_mass_allowed_in_aperture_front, then the argument aperture_radius_front will have no effect)
        :param log_mass_allowed_global_back: The minimum mass to be kept everywhere in the background (if this is smaller
        than log_mass_allowed_in_aperture_back, then the argument aperture_radius_back will have no effect)
        :param interpolated_x_angle: a list of scipy.interp1d that retuns the x angular position of a ray in
        arcsec given a comoving distance
        :param interpolated_y_angle: a list of scipy.interp1d that retuns the y angular position of a ray in
        arcsec given a comoving distance
        :param zmin: only keep halos at z > zmin
        :param zmax: only keep halos at z < zmax
        :param aperture_units: either 'ANGLES' or 'MPC'

        - If 'ANGLES', then halos are kept inside angular apertures
        around each light ray with size aperture_radius_front/aperture_radius_back.
        - If 'MPC', then halos are kept inside circular apertures with radius
        R = aperture_radius_front/back * mpc_per_arcsec(0.5)
        where D_C(0.5) is the comoving transverse distance at z = 0.5. The unit of R is arcsec * Mpc

        'ANGLES' is more conservative in that it keeps more halos in the lens model; the rendering area is basically a cone
        since the aperture size is a fixed angle at every redshift, whereas 'MPC' distributes halos in cylindrical
        tubes around each light ray along the line of sight.

        :return: A new instance of Realization with the cuts on position and mass applied
        """
        halos = []

        if zmax is None:
            zmax = self._zsource
        if zmin is None:
            zmin = 0

        for plane_index, zi in enumerate(self.unique_redshifts):

            plane_halos, _ = self.halos_at_z(zi)

            inds_at_z = np.where(self.redshifts == zi)[0]
            x_at_z = self.x[inds_at_z]
            y_at_z = self.y[inds_at_z]
            masses_at_z = np.absolute(self.masses[inds_at_z])

            if zi < zmin:
                continue
            if zi > zmax:
                continue

            comoving_distance_z = self.lens_cosmo.cosmo.D_C_z(zi)

            if zi <= self._zlens:

                minimum_mass_everywhere = deepcopy(log_mass_allowed_global_front)
                minimum_mass_in_window = deepcopy(log_mass_allowed_in_aperture_front)
                aperture_radius_arcsec = deepcopy(aperture_radius_front)

            else:

                minimum_mass_everywhere = deepcopy(log_mass_allowed_global_back)
                minimum_mass_in_window = deepcopy(log_mass_allowed_in_aperture_back)
                aperture_radius_arcsec = deepcopy(aperture_radius_back)

            keep_inds_mass = np.where(masses_at_z >= 10 ** minimum_mass_everywhere)[0]
            inds_m_low = np.where(masses_at_z < 10 ** minimum_mass_everywhere)[0]
            keep_inds_dr = []

            for idx in inds_m_low:

                for k, (interp_x, interp_y) in enumerate(zip(interpolated_x_angle, interpolated_y_angle)):

                    dx = x_at_z[idx] - interp_x(comoving_distance_z)
                    dy = y_at_z[idx] - interp_y(comoving_distance_z)

                    if aperture_units == 'ANGLES':
                        dr_cut = aperture_radius_arcsec

                    elif aperture_units == 'MPC':
                        dx *= comoving_distance_z
                        dy *= comoving_distance_z
                        dr_cut = aperture_radius_arcsec * self.lens_cosmo.cosmo.D_C_z(0.5)
                    else:
                        raise Exception('aperture units must be either MPC or ANGLES')

                    dr = np.sqrt(dx ** 2 + dy ** 2)

                    if dr <= dr_cut:
                        keep_inds_dr.append(idx)
                        break

            keep_inds = np.append(keep_inds_mass, np.array(keep_inds_dr)).astype(int)

            tempmasses = np.absolute(masses_at_z[keep_inds])

            keep_inds = keep_inds[np.where(tempmasses >= 10 ** minimum_mass_in_window)[0]]

            for halo_index in keep_inds:
                halos.append(plane_halos[halo_index])


        return Realization.from_halos(halos, self.lens_cosmo, self._prof_params,
                                      self.apply_mass_sheet_correction, self.rendering_classes,
                                      self._rendering_center_x, self._rendering_center_y, self.geometry)

    def set_rendering_classes(self, rendering_classes):

        """
        This method sets the rendering classes for the realization, which are used to apply the negative convergence sheet
        corrections after adding halos. The properties of the convergence sheets you need to add depend on the form of the
        mass function you have specified, so this information is stored in the classes used to render halos
        (LOSPowerLaw, MainLensPowerLaw, etc, see classes in pyHalo/Rendering)

        If the rendering classes are not specified for whatever reason, the code will still run but no negative convergence
        sheets will be included in your lens models. This could potentially bias results as you've effectively made every
        light cone overdense relative to the mean matter density in the Universe.

        :param rendering_classes: a list or an instance of a rendering class (LOSPowerLaw, MainLensPowerLaw)
        """

        if not isinstance(rendering_classes, list):
            rendering_classes = [rendering_classes]
        self.rendering_classes = rendering_classes

    def join(self, real, join_rendering_classes=False):

        """
        This routine combines one realization with another realization, keeping only the unqiue halos
        (as identified by their .unique_tag attribute) between them.

        :param real: another realization, possibly a filtered version of self
        :param join_rendering_classes: If True, the rendering classes associated with the new
        realization will include both the rendering class associated with self and that of real.

        :return: a new realization that contains all unique halos from self and real
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

        if join_rendering_classes:
            rendering_class_self = self.rendering_classes
            rendering_class_new = real.rendering_classes
            rendering_classes = rendering_class_self + rendering_class_new
        else:
            rendering_classes = self.rendering_classes

        centerx, centery = self.rendering_center
        return Realization.from_halos(halos, self.lens_cosmo, self._prof_params,
                                      self.apply_mass_sheet_correction, rendering_classes,
                                      centerx, centery, self.geometry)

    def shift_background_to_source(self, ray_interp_x, ray_interp_y):

        """
        This routine shifts the entire relation along a path specified by ray_interp_x/y. This routine is intended
        to be used in situations where the source is significantly offset from the origin, and you want to align the
        center of the rendering volume such that tracks the path of the light.

        :param ray_interp_x: instance of scipy.interp1d, returns the angular position of a ray
        fired through the lens center given a comoving distance
        :param ray_interp_y: same but for the y coordinate
        :return:
        """

        if self._has_been_shifted:
            return self

        halos = []

        for halo in self.halos:

            if halo.fixed_position:
                halos.append(halo)
                continue

            comoving_distance_z = self.lens_cosmo.cosmo.D_C_z(halo.z)

            xshift, yshift = ray_interp_x(comoving_distance_z), ray_interp_y(comoving_distance_z)

            halo.x += xshift
            halo.y += yshift
            halos.append(halo)

        new_realization = Realization.from_halos(halos, self.lens_cosmo, self._prof_params, self.apply_mass_sheet_correction,
                                                 self.rendering_classes, ray_interp_x, ray_interp_y, self.geometry)

        new_realization._has_been_shifted = True

        return new_realization

    def lensing_quantities(self, add_mass_sheet_correction=True, z_mass_sheet_max=None,
                           kwargs_mass_sheet_correction=None):

        """
        :param add_mass_sheet_correction: include sheets of negative convergence to correct for mass added subhalos/field halos
        :param z_mass_sheet_max: don't include negative convergence sheets at z>z_mass_sheet_max (this does nothing
        if the previous argument is False
        :pararm kwargs_mass_sheet_correction: additional keyword arguments for the mass sheet correction
        (changes the default setting)
        :return: the lens_model_list, redshift_list, kwargs_lens, and numerical_alpha_class keywords that can be plugged
        directly into a lenstronomy LensModel class
        """

        kwargs_lens = []
        lens_model_list = []
        redshift_array = []
        numerical_interp = None

        for i, halo in enumerate(self.halos):

            lens_model_name = halo.lenstronomy_ID
            kwargs_halo, interp_class = halo.lenstronomy_params
            lens_model_list += lens_model_name
            kwargs_lens += kwargs_halo
            redshift_array += [halo.z] * len(lens_model_name)

            if interp_class is not None:
                numerical_interp = interp_class

        if self.apply_mass_sheet_correction and add_mass_sheet_correction:

            if self.rendering_classes is None:
                raise Exception('if applying a convergence sheet correction, must specify '
                                'the rendering classes.')

            kwargs_mass_sheets, profile_list, z_sheets = self._mass_sheet_correction(self.rendering_classes,
                                                                                     z_mass_sheet_max,
                                                                                     kwargs_mass_sheet_correction)
            kwargs_lens += kwargs_mass_sheets
            lens_model_list += profile_list
            redshift_array = np.append(redshift_array, z_sheets)

        return lens_model_list, redshift_array, kwargs_lens, numerical_interp

    def split_at_z(self, z):
        """
        Splits the realization at redshift z, returning one instance at Realization containing all halos with
        redshift < zlens and another with all halos at redshift >= z. Be careful with the mass sheet corrections contained
        in the rendering_classes, as both new realizations will get all rendering classes from the parent realization.

        :param z: the redshift at which to split the realization
        :return: two instances at Realization divided at redshift z
        """

        halos_1, halos_2 = [], []
        for halo in self.halos:
            if halo.z <= z:
                halos_1.append(halo)
            else:
                halos_2.append(halo)

        centerx, centery = self.rendering_center
        realization_1 = Realization.from_halos(halos_1, self.lens_cosmo,
                                               self._prof_params, self.apply_mass_sheet_correction, self.rendering_classes,
                                               centerx, centery, self.geometry)
        realization_2 = Realization.from_halos(halos_2, self.lens_cosmo,
                                               self._prof_params, self.apply_mass_sheet_correction, self.rendering_classes,
                                               centerx, centery, self.geometry)

        return realization_1, realization_2

    def halo_comoving_coordinates(self):

        """
        :param halos: a list of halos
        :return: the comoving (x, y) position, mass, and redshift of each halo in the realization
        """
        xcoords, ycoords, masses, redshifts = [], [], [], []

        for halo in self.halos:
            D = self.lens_cosmo.cosmo.D_C_z(halo.z)
            x_arcsec, y_arcsec = halo.x, halo.y
            x_comoving, y_comoving = D * x_arcsec, D * y_arcsec
            xcoords.append(x_comoving)
            ycoords.append(y_comoving)
            masses.append(halo.mass)
            redshifts.append(halo.z)

        return np.array(xcoords), np.array(ycoords), np.log10(masses), np.array(redshifts)

    def halos_at_z(self, z):
        """

        :param z: redshift
        :return: all halos in the realization that are at redshift z
        """
        halos = []
        index = []
        for i, halo in enumerate(self.halos):
            if halo.z != z:
                continue
            halos.append(halo)
            index.append(i)

        return halos, index

    def mass_at_z_exact(self, z):

        """
        Computes the total mass rendered at each redshift z
        :param z: redshift
        :return: total mass rendered at z
        """

        inds = np.where(self.redshifts == z)
        m_exact = np.sum(self.masses[inds])
        return m_exact

    def number_of_halos_before_redshift(self, z):

        """
        Computes the number of halos with redshifts < z
        :param z: redshift
        :return: number of halos with redshift < z
        """
        n = 0
        for halo in self.halos:
            if halo.z < z:
                n += 1
        return n

    def number_of_halos_after_redshift(self, z):

        """
        Computes the number of halos with redshifts > z
        :param z: redshift
        :return: number of halos with redshift > z
        """
        n = 0
        for halo in self.halos:
            if halo.z > z:
                n += 1
        return n

    def number_of_halos_at_redshift(self, z):

        """
        Computes the number of halos with redshifts == z
        :param z: redshift
        :return: number of halos at z
        """

        n = 0
        for halo in self.halos:
            if halo.z == z:
                n += 1
        return n

    def _mass_sheet_correction(self, rendering_classes, z_mass_sheet_max, kwargs_mass_sheet_correction):

        """
        This routine adds the negative mass sheet corrections along the LOS and in the main lens plane.
        The actual physics that determines the amount of negative convergence to add is encoded in the rendering_classes
        (see for example Rendering.Field.PowerLaw.powerlaw_base.py)

        :param rendering_classes: the rendering class associated with each realization
        :param z_mass_sheet_max: don't add mass sheets at lens planes with redshift > z_mass_sheet_max
        :param kwargs_mass_sheet_correction: keyword arguments for the convergence sheet correction
        :return: the kwargs_lens, lens_model_list, and redshift_list of the mass sheets that can be plugged into lenstronomy
        """

        kwargs_mass_sheets = []

        redshifts = []

        profiles = []

        if self._prof_params['subtract_exact_mass_sheets']:

            for zi in self.unique_redshifts:
                area = self.geometry.angle_to_physical_area(0.5 * self.geometry.cone_opening_angle, zi)
                kwargs_mass_sheets += [{'kappa_ext': -self.mass_at_z_exact(zi) / self.lens_cosmo.sigma_crit_mass(zi, area)}]

            redshifts = self.unique_redshifts

            profiles = ['CONVERGENCE'] * len(kwargs_mass_sheets)

        else:

            for rendering_class in rendering_classes:

                if rendering_class is None:
                    continue

                kwargs_new, profiles_new, redshifts_new = \
                    rendering_class.convergence_sheet_correction(kwargs_mass_sheet_correction)

                kwargs_mass_sheets += kwargs_new
                redshifts += redshifts_new
                profiles += profiles_new

        if z_mass_sheet_max is not None:
            kwargs_mass_sheets_out = []
            profiles_out = []
            redshifts_out = []

            inds_keep = np.where(np.array(redshifts) <= z_mass_sheet_max)[0]

            for i in range(0, len(kwargs_mass_sheets)):
                if i in inds_keep:
                    kwargs_mass_sheets_out.append(kwargs_mass_sheets[i])
                    profiles_out.append(profiles[i])
                    redshifts_out.append(redshifts[i])
        else:
            kwargs_mass_sheets_out, profiles_out, redshifts_out = kwargs_mass_sheets, profiles, redshifts

        # define the center of mass sheet to be the center of the rendering volume
        centerx_interp, centery_interp = self.rendering_center

        for i, (zi, profile_name) in enumerate(zip(redshifts_out, profiles_out)):
            di = self.lens_cosmo.cosmo.D_C_z(zi)
            x_center, y_center = centerx_interp(di), centery_interp(di)
            if profile_name == 'CONVERGENCE':
                kwargs_mass_sheets_out[i]['ra_0'] = float(x_center)
                kwargs_mass_sheets_out[i]['dec_0'] = float(y_center)
            else:
                kwargs_mass_sheets_out[i]['center_x'] = float(x_center)
                kwargs_mass_sheets_out[i]['center_y'] = float(y_center)

        return kwargs_mass_sheets_out, profiles_out, redshifts_out

    def _load_halo_model(self, mass, x, y, r3d, mdef, z, is_subhalo,
                         lens_cosmo_instance, args, unique_tag):

        """
        Loads the halo model for each object based on the mass definition
        :param halo: an instance of Halo
        :return: the specific Halo class corresponding to mass definition mdef
        """

        if mdef == 'NFW':

            if is_subhalo:
                model = NFWSubhhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                    lens_cosmo_instance, args, unique_tag)
            else:
                model = NFWFieldHalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                    lens_cosmo_instance, args, unique_tag)


        elif mdef == 'TNFW':

            if is_subhalo:
                model = TNFWSubhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                    lens_cosmo_instance, args, unique_tag)

            else:
                model = TNFWFieldHalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                      lens_cosmo_instance, args, unique_tag)

        elif mdef == 'PT_MASS':

            model = PTMass(mass, x, y, r3d, mdef, z, is_subhalo,
                           lens_cosmo_instance, args, unique_tag)

        elif mdef == 'PJAFFE':

            model = PJaffeSubhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)

        elif mdef == 'coreTNFW':

            if is_subhalo:
                model = coreTNFWSubhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)
            else:
                model = coreTNFWFieldHalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)

        elif mdef == 'coreNFW':

            if is_subhalo:
                model = coreNFWSubhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)
            else:
                model = coreNFWFieldHalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)

        elif mdef == 'ULDM':

            if is_subhalo:
                model = ULDMSubhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)
            else:
                model = ULDMFieldHalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)

        elif mdef == 'GAUSSIAN_KAPPA':

            model = Gaussian(mass, x, y, r3d, mdef, z, is_subhalo,
                                  lens_cosmo_instance, args, unique_tag)

        elif mdef == 'GNFW':

            if is_subhalo:
                model = GeneralNFWSubhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                    lens_cosmo_instance, args, unique_tag)

            else:
                model = GeneralNFWFieldHalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                      lens_cosmo_instance, args, unique_tag)

        elif mdef == 'SPL_CORE':

            if is_subhalo:
                model = PowerLawSubhalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                    lens_cosmo_instance, args, unique_tag)

            else:
                model = PowerLawFieldHalo(mass, x, y, r3d, mdef, z, is_subhalo,
                                      lens_cosmo_instance, args, unique_tag)


        else:
            raise ValueError('halo profile ' + str(mdef) + ' not recongnized.')

        return model

    def _tags(self, halos=None):

        """

        :param halos: a list of halos
        :return: the unique tag for each halo in halos; if halos is not specified, returns the unique tag for each
        halo in the realization
        """
        if halos is None:
            halos = self.halos
        tags = []

        for halo in halos:

            tags.append(halo.unique_tag)

        return tags

    def _reset(self):
        """
        Resets all class attributes to the current set of halos contained in the realization
        :return:
        """

        self.x = []
        self.y = []
        self.masses = []
        self.redshifts = []
        self.r3d = []
        self.mdefs = []
        self._halo_tags = []
        self.subhalo_flags = []

        for halo in self.halos:
            self.masses.append(halo.mass)
            self.x.append(halo.x)
            self.y.append(halo.y)
            self.redshifts.append(halo.z)
            self.r3d.append(halo.r3d)
            self.mdefs.append(halo.mdef)
            self._halo_tags.append(halo.unique_tag)
            self.subhalo_flags.append(halo.is_subhalo)

        self.masses = np.array(self.masses)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.r3d = np.array(self.r3d)
        self.redshifts = np.array(self.redshifts)

        self.unique_redshifts = np.sort(np.unique(self.redshifts))

    def __eq__(self, other_reealization):

        """
        Defintes equality between two realizations if they contain the same halos with the same unique tags
        :param other_reealization:
        :return:
        """
        tags = self._tags(self.halos)
        other_tags = other_reealization._tags()
        for tag in other_tags:
            if tag not in tags:
                return False
        else:
            return True

class SingleHalo(Realization):

    def __init__(self, halo_mass, x, y, mdef, z, zlens, zsource, r3d=None, subhalo_flag=False,
                 kwargs_halo={}, cosmo=None):

        """
       Useful for generating a realization with a single or a few
        user-specified halos.
        :param halo_mass: mass of the halo in M_sun
        :param x: halo x coordinate in arcsec
        :param y: halo y coordinate in arcsec
        :param mdef: halo mass definition
        :param z: halo redshift
        :param zlens: main deflector redshift
        :param zsource: source redshift
        :param r3d: three dimensional coordinate of halo inside the host in kpc
        (only relevant for tidally-truncated subhalos, for field halos this can be None)
        :param subhalo_flag: bool, sets whether or not a halo is a subhalo
        :param kwargs_halo: keyword arguments for the halo
        :param cosmo: an instance of Cosmology(); if none is provided a default cosmology will be used
        """
        if cosmo is None:
            cosmo = Cosmology()

        lens_cosmo = LensCosmo(zlens, zsource, cosmo)

        # these are redundant keywords for a single halo, but we need to specify them anyways
        kwargs_halo.update({'cone_opening_angle': 6., 'log_mlow': 6., 'log_mhigh': 10.})
        super(SingleHalo, self).__init__([halo_mass], [x], [y],
                                         [r3d], [mdef], [z], [subhalo_flag], lens_cosmo,
                                         kwargs_realization=kwargs_halo, mass_sheet_correction=False)
