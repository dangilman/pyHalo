import numpy as np

def render_los(rendering_class, lens_plane_redshifts, delta_zs, zmin, zmax):

    redshifts = []

    init = True

    for i, (zi, delta_zi) in enumerate(zip(lens_plane_redshifts, delta_zs)):

        if zi < zmin or zi > zmax:
            continue

        m = rendering_class.render_masses(zi, delta_zi, None)

        xshift, yshift = 0., 0.
        x, y, r3 = rendering_class.render_positions_at_z(zi, len(m), xshift, yshift)

        redshifts += [zi] * len(x)

        if init:
            masses = m
            x_arcsec, y_arcsec, r3d = x, y, r3
            init = False
        else:
            x_arcsec = np.append(x_arcsec, x)
            y_arcsec = np.append(y_arcsec, y)
            masses = np.append(masses, m)
            r3d = np.append(r3d, r3)

    return masses, x_arcsec, y_arcsec, r3d, np.array(redshifts)

def render_main(rendering_class):

    masses, x_arcsec, y_arcsec, r3d, redshifts = rendering_class()

    return masses, x_arcsec, y_arcsec, r3d, redshifts
