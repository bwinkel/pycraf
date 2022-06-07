#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from enum import Enum
from collections import namedtuple
from astropy import units as apu
import numpy as np

from .helper import _Qinv
from .. import conversions as cnv
from .. import utils
# import ipdb


__all__ = [
    'building_entry_loss', 'BuildingType',
    ]


class BuildingType(Enum):
    TRADITIONAL = 1  # P.2109 "Traditional"
    THERM_EFF = 2  # P.2109 "Thermally efficient"


_BEL_COEFFS = namedtuple('_BEL_COEFFS', 'r s t u v w x y z')
_TRAD_COEFFS = _BEL_COEFFS(
    12.64, 3.72, 0.96, 9.6, 2.0, 9.1, -3.0, 4.5, -2.0,
    )
_THERM_EFF_COEFFS = _BEL_COEFFS(
    28.19, -3.00, 8.48, 13.5, 3.8, 27.8, -2.9, 9.4, -2.1,
    )


def _building_entry_loss_p2109(
        freq_ghz,
        theta_deg,
        prob,
        building_type,
        ):

    if not isinstance(building_type, BuildingType):
        raise TypeError('building_type must be a "BuildingType"')

    if not building_type in [
            BuildingType.TRADITIONAL, BuildingType.THERM_EFF
            ]:
        raise ValueError(
            'building_type must be a either '
            '"BuildingType.TRADITIONAL" or "BuildingType.THERM_EFF"'
            )

    if building_type == BuildingType.TRADITIONAL:
        c = _TRAD_COEFFS
    elif building_type == BuildingType.THERM_EFF:
        c = _THERM_EFF_COEFFS

    logf = np.log10(freq_ghz)
    qinv = _Qinv(1 - prob)

    L_h = c.r + c.s * logf + c.t * logf * logf
    L_e = 0.212 * np.abs(theta_deg)

    C = -3.
    mu1 = L_h + L_e
    mu2 = c.w + c.x * logf
    sig1 = c.u + c.v * logf
    sig2 = c.y + c.z * logf

    A = qinv * sig1 + mu1
    B = qinv * sig2 + mu2

    L_bel = 10 * np.log10(
        np.power(10, 0.1 * A) +
        np.power(10, 0.1 * B) +
        np.power(10, 0.1 * C)
        )

    return L_bel


@utils.ranged_quantity_input(
    freq=(0.08, 100, apu.GHz),
    theta=(-90, 90, apu.deg),
    prob=(0, 1, cnv.dimless),
    strip_input_units=True, output_unit=cnv.dB
    )
def building_entry_loss(
        freq,
        theta,
        prob,
        building_type,
        ):
    '''
    Calculate building entry loss (BEL).

    The BEL model is according to `Rec. ITU-R P.2109-1
    <https://www.itu.int/rec/R-REC-P.2109-1-201908-I/en>`__.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    theta : `~astropy.units.Quantity`
        Path elevation angle (w.r.t. horizon) [deg]
        The minimal loss happens at zero horizontal incidence (0 deg)
    prob : `~astropy.units.Quantity`
        Probability that loss is not exceeded [%]
    building_type : BuildingType enum
        Building type
        allowed values: `BuildingType.TRADITIONAL` or `BuildingType.THERM_EFF`

    Returns
    -------
    L_bel : `~astropy.units.Quantity`
        Building entry loss [dB]

    Examples
    --------
    With the following, one can create the Figures in `Rec. ITU-R P.2109-1
    <https://www.itu.int/rec/R-REC-P.2109-1-201908-I/en>`__.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from pycraf import conversions as cnv
        from pycraf import pathprof
        from astropy import units as u

        freq = np.logspace(-1, 2, 100) * u.GHz
        theta = 0 * u.deg
        prob = 0.5 * cnv.dimless

        plt.figure(figsize=(8, 6))

        for btype in [
                pathprof.BuildingType.TRADITIONAL,
                pathprof.BuildingType.THERM_EFF
                ]:

            L_bel = pathprof.building_entry_loss(freq, theta, prob, btype)

            plt.semilogx(freq, L_bel, '-', label=str(btype))

        plt.xlabel('Frequency [GHz]')
        plt.ylabel('BEL [dB]')
        plt.xlim((0.1, 100))
        plt.ylim((10, 60))
        plt.legend(*plt.gca().get_legend_handles_labels())
        plt.title('Median BEL at horizontal incidence')
        plt.grid()

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from pycraf import conversions as cnv
        from pycraf import pathprof
        from astropy import units as u

        freqs = [0.1, 1, 10, 100] * u.GHz
        colors = ['k', 'r', 'g', 'b']
        theta = 0 * u.deg
        prob = np.linspace(1e-6, 1 - 1e-6, 200) * cnv.dimless

        plt.figure(figsize=(8, 6))

        for freq, color in zip(freqs, colors):

            for btype, ls in zip([
                    pathprof.BuildingType.TRADITIONAL,
                    pathprof.BuildingType.THERM_EFF
                    ], ['--', ':']):

                L_bel = pathprof.building_entry_loss(freq, theta, prob, btype)

                plt.plot(L_bel, prob, ls, color=color)

            # labels
            plt.plot([], [], '-', color=color, label=str(freq))

        plt.xlabel('BEL [dB]')
        plt.ylabel('Probability')
        plt.xlim((-20, 140))
        plt.ylim((0, 1))
        plt.legend(*plt.gca().get_legend_handles_labels())
        plt.title('BEL (dashed: Traditional, dotted: Thermally inefficient')
        plt.grid()

    Notes
    -----
    - The result of this function is to be understood as a cumulative
      value. For example, if `prob = 2%`, it means that for
      2% of all possible outcomes, the loss will not exceed the
      returned `L_bel` value, for the remaining 98% of locations it
      will therefore be lower than `L_bel`. The smaller `prob`,
      the smaller the returned `L_bel`, i.e., low BELs
      are more unlikely.
    '''

    return _building_entry_loss_p2109(
        freq,
        theta,
        prob,
        building_type,
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
