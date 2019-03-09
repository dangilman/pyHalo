import numpy as np
import matplotlib.pyplot as plt


def classical_semi(sigma0=None, alpha=None, v=None, v_pivot=10):

    x = v * v_pivot ** -1

    return sigma0 * x ** alpha

def classical_cored_1(sigma0=None, alpha=None, v=None, v_pivot=10, v_crit = 1, nu=2):

    x1 = v * v_crit ** -1
    x2 = v * v_pivot ** -1

    return sigma0 * ((1 + x2 ** nu) * (1 + x1**nu)) ** (0.5 * alpha / nu)

def classical_cored_2(sigma0=None, alpha=None, v=None, v_pivot=10, v_crit = 1, nu = 2):

    screen_factor = v_crit * v_pivot**-1
    x1 = v * v_pivot ** -1

    return sigma0 * ((screen_factor ** nu + x1 ** nu)) ** (alpha*nu**-1)
