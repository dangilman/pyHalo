from pyHalo.Spatial.nfw import NFW3DFast
import numpy as np

class NFW3DCore(object):

    def __init__(self, Rs, rmax2d, rmax3d, r_core_parent):

        self._Rs = Rs
        self._rmax2d = rmax2d
        self._rmax3d = rmax3d
        self._x3dmax = rmax3d / Rs

        self._xcore = r_core_parent / Rs

        self.nfw = NFW3DFast(Rs, rmax2d, rmax3d)

        self._xmin = self.nfw._xmin

        self._norm = ((self._xmin + self._xcore) * (1 + self._xmin) ** 2) ** -1

    def _eval_rho_core(self, x, xcore):

        x = max(self._xmin, x)
        rho = ((x + xcore) * (1 + x) ** 2) ** -1

        return rho/self._norm

    def p_x(self, x, xcore):

        p1 = self._eval_rho_core(x, xcore)
        p2 = self._eval_rho_core(x, 0.)
        return p1/p2

    def _draw(self, N, zlens):

        x, y, r2, r3 = self.nfw.draw(N, zlens)
        x3 = r3/self._Rs
        prob = np.array([self.p_x(x3i, self._xcore) for x3i in x3])
        u = np.random.rand(N)
        keep = np.where(u < prob)[0]

        return x[keep], y[keep], r2[keep], r3[keep]

    def draw(self, N, zlens):

        x, y, r2, r3 = self._draw(N, zlens)

        while len(x) < N:
            _x, _y, _r2, _r3 = self._draw(N-len(x), zlens)
            x = np.append(x, _x)
            y = np.append(y, _y)
            r2 = np.append(r2, _r2)
            r3 = np.append(r3, _r3)

        return x, y, r2, r3


