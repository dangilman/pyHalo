import numpy as np

def render(rendering_class, lens_plane_redshifts, delta_zs):

    redshifts = []

    for i, (zi, delta_zi) in enumerate(zip(lens_plane_redshifts, delta_zs)):

        m = rendering_class.render_masses(zi, delta_zi)

        rescale_angle = rendering_class.rescale_angle(zi, delta_zi)
        x, y, r2, r3 = rendering_class.render_positions_at_z(zi, len(m), rescale_angle)

        redshifts += [zi] * len(x)

        if i == 0:
            masses = m
            x_arcsec, y_arcsec, r2d, r3d = x, y, r2, r3
        else:
            x_arcsec = np.append(x_arcsec, x)
            y_arcsec = np.append(y_arcsec, y)
            masses = np.append(masses, m)
            r2d = np.append(r2d, r2)
            r3d = np.append(r3d, r3)

    return masses, x_arcsec, y_arcsec, r2d, r3d, np.array(redshifts)
