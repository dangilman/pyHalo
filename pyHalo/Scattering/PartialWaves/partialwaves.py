import numpy as np
from pyHalo.Scattering.PartialWaves.interactions import YukawaInteraction, NoInteraction
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class PartialWaves(object):

    def __init__(self, potential, potential_args):

        self._potential = potential

        if potential == 'Yukawa':
            self.interaction_potential = YukawaInteraction(**potential_args)
            self._solver = self._yukawa_solver
        elif potential == 'None':
            self.interaction_potential = NoInteraction(**potential_args)
            self._solver = self._plane_wave_solver

    @property
    def properties(self):
        return self.interaction_potential.properties

    def eval_cross(self, tol=0.01):

        lstep = 5
        lmax = 0
        cross = []
        counter = 0

        while lmax < 500:
            ci = self._solve_iterative(lmax)
            cross.append(ci)
            if counter > 5:
                if np.std(cross[-5:]) < tol * np.mean(cross[-5:]):
                    break
            counter += 1
            lmax += lstep

            if counter%lstep == 0:
                print(lmax)

        return cross

    def _solve_iterative(self, l_max, tol=0.05):

        count = 0
        cross = 0
        n_scale = 15
        scale1, scale2 = 1, 2
        for l in range(0, l_max + 1):
            cross1 = self._solve_iterative_single(l, scale=1)
            cross2 = self._solve_iterative_single(l, scale=2)

            if abs(cross1 - cross2) * cross2 ** -1 < tol or count > n_scale:
                cross = 0.5*(cross1 + cross2)
                break
            else:
                scale1 *= 1.5
                scale2 *= 1.5
                count+=1

        return cross

    def _solve_iterative_single(self, l_max, scale=1):

        cross = 0
        for l in range(0, l_max+1):
            chi, chi_prime, x = self.solve_schrodinger_equation_chi(l, scale=scale)
            #maxx = max(chi)
            #chi *= maxx ** -1
            #chi_prime *= maxx ** -1
            delta_l = self.interaction_potential.phase_shift(l, x[-1], chi_prime[-1], chi[-1])
            cross += np.sin(delta_l**2) * (2*l + 1)

        return 4*np.pi*cross / self.interaction_potential.v**2

    def _x_init(self, scale=1):

        x_min = self.interaction_potential.x_min_init
        x_max = self.interaction_potential.x_max_init

        x_min = max(0.001, x_min * scale ** -0.25)
        return x_min, x_max * scale ** 1.25

    def _yukawa_solver(self, l, x_max, init_scale=1):

        x_min, _x_max = self._x_init(init_scale)

        if x_max is None:
            x_max = _x_max
        U0 = [1, (l+1) / x_min]
        x = np.linspace(x_min, x_max, 150)

        U = odeint(self.interaction_potential, U0, x, args=(l,))

        return U[:,0], U[:,1], x

    def solve_schrodinger_equation(self, l, xmax=None, scale=1):

        chi, chi_prime, x = self.solve_schrodinger_equation_chi(l, xmax=xmax, scale=scale)

        return chi / x, chi_prime / x, x

    def solve_schrodinger_equation_chi(self, l, xmax=None, scale=1):

        chi, chi_prime, x = self._solver(l, xmax, scale)

        return chi, chi_prime, x

if False:
    p = PartialWaves('Yukawa', {'m_phi': 4, 'm_chi': 1, 'coupling': -3, 'v': 340})
    c = p.eval_cross(tol=0.1)
    plt.plot(c); plt.show()
    print(c[-1])

if True:
    cross = []
    v = np.logspace(0, np.log10(30), 25)

    for vi in v:
        print(vi)
        p = PartialWaves('Yukawa', {'m_phi': 100, 'm_chi': 1, 'coupling': -0.05, 'v': vi})
        print(p.properties)
        c = p.eval_cross()
        cross.append(c[-1])
    cross = np.array(cross)
    logv = np.log10(v)
    logc = np.log10(cross)
    plt.scatter(logv, logc)
    norm = cross[0] * v[0] ** 0.6
    #plt.loglog(v, norm*v ** -0.6)
    print(np.polyfit(logv, logc, 1))

    plt.show()
