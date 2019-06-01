import numpy as np
from scipy.interpolate import interp1d

class NFW_2D(object):

    def __init__(self, Rs, rmax2d, xoffset=0, yoffset = 0):

        """
        all distances expressed in (physical) kpc

        :param Rs: scale radius
        :param rmax2d: maximum 2d radius
        :param rmax3d: maximum 3d radius (basically sets the distribution of z coordinates)
        :param xoffset: centroid of NFW
        :param yoffset: centroid of NFW

        """

        self.rmax2d = rmax2d
        self.rs = Rs

        self.xoffset = xoffset
        self.yoffset = yoffset

        self.xoffset,self.yoffset = xoffset,yoffset

        self.xmin = 0.001

    def _density_2d(self, x):

        if x < 1:
            f = np.sqrt(1-x**2)
            func = np.arctanh(f) * f ** -1
        elif x > 1:
            f = np.sqrt(-1 + x ** 2)
            func = np.arctan(f) * f ** -1

        return 2 * (1 - func) * (x**2 - 1)

    def draw(self, N):

        x, y, r2d = [], [], []
        rho2d_max = self._density_2d(self.xmin)

        while len(r2d) < N:

            theta = np.random.uniform(0, 2 * np.pi)

            r_2 = np.random.uniform(0, self.rmax2d ** 2) ** 0.5

            _x = r_2 * self.rs ** -1

            draw = self._density_2d(_x) * rho2d_max ** -1

            if draw > np.random.rand():
                x.append(r_2 * np.cos(theta))
                y.append(r_2 * np.sin(theta))
                r2d.append(r_2)

        # just make r3d some big number for truncation purposes
        r3d = np.ones_like(r2d) * 400
        return np.array(x) + self.xoffset, np.array(y) + self.yoffset, np.array(r2d), r3d

class NFW_3D(object):

    def __init__(self, Rs, rmax2d, rmax3d, xoffset=0, yoffset = 0, tidal_core=False, r_core_parent = None):

        """
        all distances expressed in (physical) kpc

        :param Rs: scale radius
        :param rmax2d: maximum 2d radius
        :param rmax3d: maximum 3d radius (basically sets the distribution of z coordinates)
        :param xoffset: centroid of NFW
        :param yoffset: centroid of NFW
        :param tidal_core: flag to draw from a uniform denity inside r_core
        :param r_core: format 'number * Rs' where number is a float.
        specifies an inner radius where the distribution is uniform
        see Figure 4 in Jiang+van den Bosch 2016
        """

        self.rmax3d = rmax3d
        self.rmax2d = rmax2d
        self.rs = Rs

        self.xoffset = xoffset
        self.yoffset = yoffset

        rmin = Rs*0.005

        self.xoffset,self.yoffset = xoffset,yoffset
        self.tidal_core = tidal_core

        self.r_core = r_core_parent

        self.xmin = rmin * Rs ** -1
        self.xmax = rmax3d * Rs ** -1

    def draw(self,N):

        r3d, x, y, r2d,z = [], [], [], [],[]

        while len(r3d) < N:

            theta = np.random.uniform(0,2*np.pi)
            phi = np.random.uniform(0,2*np.pi)

            r2 = np.random.uniform(0,self.rmax2d**2) ** 0.5

            if r2 > self.rmax2d:
                continue

            r_z = np.random.uniform(0,self.rmax3d**2) ** 0.5

            x_value,y_value = r2*np.cos(theta),r2*np.sin(theta)
            z_value = r_z * np.sin(phi)

            r3 = (r2**2+z_value**2)**0.5

            if r3 > self.rmax3d:
                continue

            if self._acceptance_prob(r3) > np.random.uniform(0,1):
                r3d.append(r3)
                x.append(x_value+self.xoffset)
                y.append(y_value+self.yoffset)
                z.append(z_value)
                r2d.append(r2)

        x = np.array(x)
        y = np.array(y)
        r2d = np.array(r2d)
        r3d = np.array(r3d)

        return x,y,r2d,r3d

    def _acceptance_prob(self, r3d):

        if self.tidal_core:

            prob = self._density_3d(max(self.r_core, r3d)) * \
                   self._upper_bound()**-1
        else:
            prob = self._density_3d(r3d) * self._upper_bound() ** -1

        return prob

    def _density_3d(self, r):

        x = r*self.rs**-1

        if isinstance(x,float) or isinstance(x,int):
            x = max(self.xmin,x)
        else:
            x[np.where(x<self.xmin)] = self.xmin

        return (x*(1+x)**2)**-1

    def _upper_bound(self):

        norm = self._density_3d(self.xmin * self.rs)
        return norm


def approx_cdf_1d(x_array, pdf_array):
    """

    :param x_array: x-values of pdf
    :param pdf_array: pdf array of given x-values
    """
    norm_pdf = pdf_array / np.sum(pdf_array)
    cdf_array = np.zeros_like(norm_pdf)
    cdf_array[0] = norm_pdf[0]
    for i in range(1, len(norm_pdf)):
        cdf_array[i] = cdf_array[i - 1] + norm_pdf[i]
    cdf_func = interp1d(x_array, cdf_array)
    cdf_inv_func = interp1d(cdf_array, x_array)
    return cdf_array, cdf_func, cdf_inv_func


def rhonfw_x(x, norm=1):
    return norm * (x * (1 + x) ** 2) ** -1


def _rhonfw_tidal(x, xtidal):
    norm = rhonfw_x(xtidal)
    if x >= xtidal:
        return rhonfw_x(x)
    else:
        return norm


def rhonfw_tidal(x, xtidal):
    if isinstance(x, float) or isinstance(x, int):
        return _rhonfw_tidal(x, xtidal)
    else:
        vals = []
        for xi in x:
            vals.append(_rhonfw_tidal(xi, xtidal))
        return np.array(vals)

class NFW3DFast(object):

    def __init__(self, Rs, rmax2d, rmax3d, xoffset=0, yoffset=0, tidal_core=False, r_core_parent=None):

        """
        all distances expressed in (physical) kpc

        :param Rs: scale radius
        :param rmax2d: maximum 2d radius
        :param rmax3d: maximum 3d radius (basically sets the distribution of z coordinates)
        :param xoffset: centroid of NFW
        :param yoffset: centroid of NFW
        :param tidal_core: flag to draw from a uniform denity inside r_core
        :param r_core: format 'number * Rs' where number is a float.
        specifies an inner radius where the distribution is uniform
        see Figure 4 in Jiang+van den Bosch 2016
        """

        self.rmax3d = rmax3d
        self.rmax2d = rmax2d
        self.rs = Rs

        self.xoffset = xoffset
        self.yoffset = yoffset
        xmin, xmax = 0.001, rmax3d * Rs ** -1
        x = np.arange(xmin, xmax, 0.01)

        if tidal_core:
            self._tidal_core = True
            self._xtidal = r_core_parent * Rs ** -1
            pdf = rhonfw_tidal(x, self._xtidal)
        else:
            self._tidal_core = False
            pdf = rhonfw_x(x)

        cdf_array, _, self._cdf_inv_func = approx_cdf_1d(x, pdf)
        self._umin, self._umax = cdf_array[0], cdf_array[-1]

    def draw(self, N):

        r3d, x, y, r2d, z = [], [], [], [], []

        while len(r3d) < N:

            theta = np.random.uniform(0, 2 * np.pi)

            r2 = np.random.uniform(0, self.rmax2d ** 2) ** 0.5

            if r2 > self.rmax2d:
                continue

            x_value, y_value = r2 * np.cos(theta), r2 * np.sin(theta)
            u = np.random.uniform(self._umin, self._umax)
            z_value = self._cdf_inv_func(u) * self.rs

            _r2d = (x_value ** 2 + y_value ** 2) ** 0.5
            _r3d = np.sqrt(_r2d ** 2 + z_value ** 2)

            X1 = _r3d / self.rs
            X2 = z_value / self.rs

            if self._tidal_core:
                prob = rhonfw_tidal(X1, self._xtidal) * rhonfw_tidal(X2, self._xtidal) ** -1
            else:
                prob = rhonfw_x(X1) * rhonfw_x(X2) ** -1
            accept = prob > np.random.rand()

            if _r3d <= self.rmax3d and accept:
                x.append(x_value)
                y.append(y_value)
                r2d.append(_r2d)
                r3d.append(_r3d)

        x = np.array(x)
        y = np.array(y)
        r2d = np.array(r2d)
        r3d = np.array(r3d)

        return x, y, r2d, r3d


