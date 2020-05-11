def TNFWprofile(r, rhos, rs, rt):

    x = r/rs
    tau = rt/rs

    return tau ** 2 * rhos/(x * (1 + x) ** 2 * (x **2 + tau**2))


def SIDM_TNFWprofile(r, rhos, rs, rt, rc):

    x = r / rs
    tau = rt / rs
    beta = rc/rs

    fac = ()
    return tau ** 2 * rhos / (x * (1 + x) ** 2 * (x ** 2 + tau ** 2))
