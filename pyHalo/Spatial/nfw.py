import numpy as np
from scipy.interpolate import interp1d


class NFW3D(object):

    def __init__(self, Rs, rmax2d, rmax3d, r_core_parent=None):

        self._Rs = Rs
        self._rmax2d = rmax2d
        self._rmax3d = rmax3d
        self._x3dmax = rmax3d / Rs

        if r_core_parent is not None:
            self._xmin = r_core_parent / Rs
        else:
            self._xmin = 0.001 * Rs

        self._norm = self._eval_rho(self._xmin)
        self._x2dmax = self._rmax2d / Rs

    @staticmethod
    def _eval_rho(x):
        return (x * (1 + x) ** 2) ** -1

    def _sample_uniform_xy(self):
        theta = np.random.uniform(0, 2 * np.pi)
        x2d = np.sqrt(np.random.uniform(0, self._rmax2d ** 2))
        x_kpc, y_kpc = x2d * np.cos(theta), x2d * np.sin(theta)
        return x_kpc, y_kpc

    def _generate_proposal(self):

        x_kpc, y_kpc = self._sample_uniform_xy()

        z_kpc = self._sample_z()

        return x_kpc, y_kpc, z_kpc

    def _sample_z(self):

        z = np.random.uniform(self._xmin * self._Rs, self._x3dmax * self._Rs)
        return z

    def _density(self, x):

        constant_prob = 1
        max_x = max(self._xmin, x)
        ratio = self._eval_rho(max_x) / self._eval_rho(self._xmin)
        return constant_prob * ratio

    def _draw_single(self):

        # all in kpc
        while True:
            xprop, yprop, zprop = self._generate_proposal()

            r2dprop = (xprop ** 2 + yprop ** 2) ** 0.5

            r3dprop = (r2dprop ** 2 + zprop ** 2) ** 0.5

            X3d = r3dprop / self._Rs

            u = np.random.rand()

            prob = self._density(X3d)

            if prob >= u:
                return xprop, yprop, r2dprop, r3dprop

    def draw(self, N, zlens):

        x_kpc, y_kpc, r2d_kpc, r3d_kpc = [], [], [], []

        for i in range(0, N):
            out = self._draw_single()
            x_kpc.append(out[0])
            y_kpc.append(out[1])
            r2d_kpc.append(out[2])
            r3d_kpc.append(out[3])

        return np.array(x_kpc), np.array(y_kpc), np.array(r2d_kpc), np.array(r3d_kpc)


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
        xmin, xmax = 0.0001, rmax3d * Rs ** -1
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

    def draw(self, N, zlens):

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

            if accept and _r3d <= self.rmax3d:
                x.append(x_value)
                y.append(y_value)
                r2d.append(_r2d)
                r3d.append(_r3d)

        x = np.array(x)
        y = np.array(y)
        r2d = np.array(r2d)
        r3d = np.array(r3d)

        return x, y, r2d, r3d


