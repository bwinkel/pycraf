# -*- coding: utf-8 -*-
"""
Protection-distance assessment between inductive (loop) systems and
radiocommunication services below 30 MHz, after ITU-R SM.2028-0 (2012).

From a magnetic field strength measured at a known distance, the model
recovers the magnetic dipole moment and effective radiated power of the
interferer, then returns the separation distance at which the field drops
to the victim's permissible limit. Both victim geometries of the
Recommendation are supported: a ground-wave path (40 dB/dec then
20 dB/dec roll-off, eqs. 9-13) and a free-space path (20 dB/dec only,
eqs. 11-13). Ground electrical parameters and the 40 dB/dec asymptote
field strength are taken from ITU-R P.368 / SM.2028-0 Table 1.

Public API: :func:`findE40`, :func:`low_prop`, :func:`protection_distance`.
"""
import numpy as np
import astropy.units as apu
from astropy import constants
from .. import utils, conversions as cnv

Easy20_dB = 109.5                                # Field strength at 20 dB/dec roll-off [dB(uV/m)]
C_M_S     = constants.c.to_value(apu.m / apu.s) # Speed of light [m/s]

# Physical units carried by the Table 1 data (the arrays below are stored
# unit-less for indexing/log-domain arithmetic; these constants document and
# re-attach the units, e.g. ``TAB368_TYPES * E40_UNIT``).
E40_UNIT   = cnv.dB_uV_m       # 40 dB/dec asymptote field strength
SIGMA_UNIT = apu.S / apu.m     # ground conductivity sigma
EPS_UNIT   = cnv.dimless       # relative permittivity epsilon_r

__all__ = [
    "findE40",
    "low_prop",
    "protection_distance",
]

TAB368_KHZ = np.array([
    10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000,
    1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 30000
    ])  # column frequencies [kHz] of ITU-R P.368 / SM.2028-0 Table 1

# 40 dB/dec asymptote field strength E_asymptote,40, in dB(uV/m) at 1 km for
# 1 kW ERP (see ``E40_UNIT``). Rows follow the ``Ground_Types`` order; columns
# follow ``TAB368_KHZ``. Stored unit-less so it can be indexed and combined in
# the log domain; multiply by ``E40_UNIT`` to obtain a Quantity.
TAB368_TYPES = np.array([
    [166,164,163,162,162,161,160,159,158,158,157,156,156,154,152,151,150,147,144,142,136,132,126,120,113],
    [166,165,164,163,162,161,160,159,158,158,157,156,155,154,153,153,152,151,149,148,146,143,138,134,127],
    [165,164,163,162,161,159,157,155,151,147,141,136,132,126,122,118,115,111,108,107,103,101, 97, 95, 91],
    [167,165,164,163,162,162,161,160,158,157,155,153,150,146,142,135,129,123,117,113,105,100, 95, 91, 87],
    [165,163,163,162,161,161,159,158,156,154,150,147,143,137,132,124,119,112,107,103, 97, 94, 89, 87, 83],
    [165,164,163,163,162,161,158,156,153,148,142,135,134,127,120,114,109,103, 99, 97, 93, 90, 87, 84, 80],
    [165,164,163,161,160,158,154,150,144,140,132,127,123,117,112,107,103, 98, 95, 93, 89, 87, 83, 81, 77],
    [164,163,162,158,155,152,146,142,134,129,122,117,113,107,103, 98, 95, 93, 90, 87, 84, 81, 77, 75, 72],
    [163,160,157,152,148,144,137,132,124,119,112,107,103, 98, 96, 92, 89, 86, 83, 81, 78, 76, 72, 70, 66],
    [159,154,149,142,137,133,126,121,115,111,107,104,102, 98, 96, 92, 89, 86, 84, 82, 78, 76, 72, 70, 66],
    [151,144,139,132,128,124,119,116,112,109,106,103,102, 98, 96, 92, 89, 86, 83, 82, 78, 76, 72, 70, 66],
    ], dtype=float)

# Ground electrical parameters, ITU Handbook on Ground Wave Propagation (2014),
# Table 2 (= ITU-R P.368 figure constants): (sigma, eps_r). Conductivity in its
# natural unit (S/m, mS/m or uS/m); permittivity dimensionless.
_mS = apu.mS / apu.m
_uS = apu.uS / apu.m
Ground_Types = {
    "sea_water_low_salinity":     (1.0  * SIGMA_UNIT, 80 * EPS_UNIT),
    "sea_water_average_salinity": (5.0  * SIGMA_UNIT, 80 * EPS_UNIT),
    "fresh_water":                (3    * _mS,        80 * EPS_UNIT),
    "land_very_wet":              (30   * _mS,        40 * EPS_UNIT),
    "wet_ground":                 (10   * _mS,        30 * EPS_UNIT),
    "land":                       (3    * _mS,        22 * EPS_UNIT),
    "medium_dry_ground":          (1    * _mS,        15 * EPS_UNIT),
    "dry_ground":                 (0.3  * _mS,         7 * EPS_UNIT),
    "very_dry_ground":            (0.1  * _mS,         3 * EPS_UNIT),
    "fresh_water_ice_-1_c":       (30   * _uS,         3 * EPS_UNIT),
    "fresh_water_ice_-10_c":      (10   * _uS,         3 * EPS_UNIT),
}

# Lookup keyed by ground type, row aligned to Ground_Types order. Sigma reduced
# to canonical S/m; sigma/eps_r stored as floats (findE40 re-attaches units).
_GT_DATA = {
    k: (TAB368_TYPES[i], sigma.to_value(SIGMA_UNIT), eps.to_value(EPS_UNIT))
    for i, (k, (sigma, eps)) in enumerate(Ground_Types.items())
}
# Bin edges midway between adjacent table frequencies: searchsorted then maps a
# frequency to its nearest tabulated column (step-wise table, not interpolated).
_TAB_MIDPT = (TAB368_KHZ[:-1] + TAB368_KHZ[1:]) / 2.0

# Allowed victim location strings (normalised)
_VICTIM_LOCATIONS = ('ground_wave', 'free_space')
_REGIMES_GW = np.array(['near-field', 'close near-field', 'GW 40dB/dec', 'GW 20dB/dec'], dtype='<U16')
_REGIMES_FS = np.array(['near-field', 'close near-field', 'FS 20dB/dec'], dtype='<U16')


@utils.ranged_quantity_input(
    freq=(0.01, 30, apu.MHz),
    strip_input_units=True,
    output_unit=(cnv.dB_uV_m, apu.S / apu.m, cnv.dimless),
)
def findE40(freq, ground_type):
    """
    40 dB/dec asymptote field strength from ITU-R P.368 / SM.2028-0 Table 1.

    Looks up E_asymptote,40 (at 1 km for 1 kW ERP) for the given ground type and
    frequency, returning it together with the ground's electrical parameters.

    Parameters
    ----------
    freq : `~astropy.units.Quantity` [MHz]
        Frequency, 0.01 to 30 MHz. Scalar or array.
    ground_type : str
        Ground type, case-insensitive; spaces and punctuation are ignored
        (``"Wet ground"`` == ``"wet_ground"``). Valid keys are listed in
        ``Ground_Types`` (e.g. ``sea_water_average_salinity``, ``land``,
        ``dry_ground``, ``fresh_water_ice_-10_c``).

    Returns
    -------
    Easy40 : `~astropy.units.Quantity` [dB(uV/m)]
        40 dB/dec asymptote field strength.
    sigma : `~astropy.units.Quantity` [S/m]
        Ground conductivity.
    eps_r : `~astropy.units.Quantity` [dimensionless]
        Relative permittivity.

    Raises
    ------
    ValueError
        If `ground_type` is not in ``Ground_Types``.
    """
    return _findE40(freq, ground_type)


@utils.ranged_quantity_input(
    Hm=(None, None, apu.dB(apu.uA / apu.m)),
    d=(0, None, apu.m),
    freq=(0.01, 30, apu.MHz),
    E_limit=(None, None, apu.dB(apu.uV / apu.m)),
    strip_input_units=True,
    output_unit=(apu.A * apu.m**2, None, apu.m, cnv.dB_W, apu.km, None),
)
def low_prop(Hm, d, freq, ground_type, E_limit, BWR=0.0,
             victim_location='ground_wave'):
    """
    ERP and protection distance from a measured H-field (ITU-R SM.2028-0).

    Recovers the magnetic dipole moment and ERP of the inductive source from a
    field measured at distance `d`, then returns the separation distance at
    which the field falls to `E_limit` for the chosen victim geometry.

    Parameters
    ----------
    Hm : `~astropy.units.Quantity` [dB(uA/m)]
        Magnetic field strength measured at `d`. Scalar or array.
    d : `~astropy.units.Quantity` [m]
        Measurement distance (> 0).
    freq : `~astropy.units.Quantity` [MHz]
        Operating frequency, 0.01 to 30 MHz. Scalar or array.
    ground_type : str
        Propagation-path ground type; see ``Ground_Types``.
    E_limit : `~astropy.units.Quantity` [dB(uV/m)]
        Permissible field strength at the victim.
    BWR : `~astropy.units.Quantity` [dB], optional
        Bandwidth ratio added to `E_limit` (eq. 22). Default 0.
    victim_location : {'ground_wave', 'free_space'}, optional
        Victim path model. Ground wave (default) uses eqs. 9, 11-13;
        free space uses eqs. 11-13.

    Returns
    -------
    m : `~astropy.units.Quantity` [A m^2]
        Magnetic dipole moment (coaxial if d < 2.354*lam_r, else coplanar).
    m_type : ndarray of str
        ``'coaxial'`` or ``'coplanar'``.
    d_transition : `~astropy.units.Quantity` [m]
        40-to-20 dB/dec transition distance (ground wave only; NaN otherwise).
    ERP : `~astropy.units.Quantity` [dBW]
        Effective radiated power.
    distance : `~astropy.units.Quantity` [km]
        Protection (separation) distance.
    regime : ndarray of str
        Regime applied: ``'near-field'``, ``'close near-field'``,
        ``'GW 40dB/dec'``/``'GW 20dB/dec'`` (ground wave) or ``'FS 20dB/dec'``.

    Raises
    ------
    ValueError
        If `victim_location` or `ground_type` is invalid.

    Notes
    -----
    `Hm`, `d`, `freq` and `E_limit` are fully vectorised: pass arrays and call
    once rather than looping in Python. Most of the per-call cost is astropy's
    unit-checking overhead, so one call over N inputs is far faster than N
    scalar calls (orders of magnitude for large N).
    """
    return _low_prop(Hm, d, freq, ground_type, E_limit, BWR, victim_location)


@utils.ranged_quantity_input(
    m=(None, None, apu.A * apu.m**2),
    ERP=(None, None, cnv.dB_W),
    Easy40=(None, None, cnv.dB_uV_m),
    E_limit=(None, None, apu.dB(apu.uV / apu.m)),
    lam_r=(0, None, apu.m),
    d_tr=(0, None, apu.m),
    strip_input_units=True,
    output_unit=(apu.m, None),
)
def protection_distance(m, ERP, Easy40, E_limit, lam_r, d_tr,
                        BWR=0.0, victim_location='ground_wave'):
    """
    Protection distance for a given propagation regime (ITU-R SM.2028-0 Sec. 4).

    Selects the applicable roll-off regime (top-down: 40 dB/dec, 20 dB/dec,
    close near-field, near-field) and returns the corresponding separation
    distance.

    Parameters
    ----------
    m : `~astropy.units.Quantity` [A m^2]
        Magnetic dipole moment.
    ERP : `~astropy.units.Quantity` [dBW]
        Effective radiated power.
    Easy40 : `~astropy.units.Quantity` [dB(uV/m)]
        40 dB/dec asymptote field strength (from ``findE40``).
    E_limit : `~astropy.units.Quantity` [dB(uV/m)]
        Permissible field strength at the victim.
    lam_r : `~astropy.units.Quantity` [m]
        Radian wavelength, lambda/(2*pi).
    d_tr : `~astropy.units.Quantity` [m]
        40-to-20 dB/dec transition distance (ground wave only).
    BWR : `~astropy.units.Quantity` [dB], optional
        Bandwidth ratio added to `E_limit`. Default 0.
    victim_location : {'ground_wave', 'free_space'}, optional
        Victim path model. Default ``'ground_wave'``.

    Returns
    -------
    distance : `~astropy.units.Quantity` [m]
        Protection (separation) distance.
    regime : ndarray of str
        Regime applied (see :func:`low_prop`).

    Raises
    ------
    ValueError
        If `victim_location` is invalid.
    """
    if hasattr(BWR, 'value'): BWR = BWR.value   # accept Quantity or plain float
    ERP_dBkW  = ERP - 30.0        # dBW -> dB(kW)
    E_eff     = E_limit + BWR     # eq. 22: E_interference = E_limit + BWR
    return _protection_distance(m, ERP_dBkW, Easy40, E_eff, lam_r, d_tr,
                                victim_location)


def _findE40(freq, ground_type):
    key = ground_type.lower()
    if key not in _GT_DATA:
        raise ValueError(
            f"Ground type '{ground_type}' not recognized. "
            f"Valid names: {list(Ground_Types)}"
        )
    row, cond_gr, permit_gr = _GT_DATA[key]
    freq_kHz = np.atleast_1d(freq) * 1e3
    idx      = np.clip(np.searchsorted(_TAB_MIDPT, freq_kHz), 0, len(TAB368_KHZ) - 1)
    E40      = row[idx]                       # row is already float64
    return (E40.item() if E40.size == 1 else E40), cond_gr, permit_gr


def _protection_distance(m, ERP_dBkW, Easy40_dB, E_limit, lam_r, d_tr,
                         victim_location='ground_wave'):
    # ERC Report 69 Sec.8 top-down priority (worst case): 40->20->close NF->inside NF
    Hl    = np.power(10.0, (E_limit - 171.5) / 20.0)              # eq. 8  [A/m]
    cr    = 2.354 * lam_r
    r_nf  = np.cbrt(m / (2.0 * np.pi * Hl))                       # eq. 13 [m]
    r_cnf = np.sqrt(m / (2.0 * np.pi * Hl * lam_r))               # eq. 12 [m]
    r_20  = np.power(10.0, (169.5 + ERP_dBkW - E_limit) / 20.0)   # eq. 11 [m]
    if victim_location == 'ground_wave':
        r_40 = 1000.0 * np.power(10.0, (Easy40_dB + ERP_dBkW - E_limit) / 40.0)  # eq. 9
        idx  = np.where((r_40 >= d_tr) & (r_40 >= cr), 2,
               np.where(r_20 >= cr, 3,
               np.where(r_cnf >= lam_r, 1, 0)))
        # elementwise pick (np.array([...])[idx] would broadcast to N x N)
        r = np.choose(idx, np.broadcast_arrays(r_nf, r_cnf, r_40, r_20))
        return r, _REGIMES_GW[idx]
    idx = np.where(r_20 >= cr, 2, np.where(r_cnf >= lam_r, 1, 0))
    r = np.choose(idx, np.broadcast_arrays(r_nf, r_cnf, r_20))
    return r, _REGIMES_FS[idx]


def _low_prop(Hm, d, freq, ground_type, E_limit, BWR=0.0, victim_location='ground_wave'):
    loc = victim_location.lower().replace(' ', '_')
    if loc not in _VICTIM_LOCATIONS:
        raise ValueError(
            f"victim_location '{victim_location}' not recognized. "
            f"Valid values: {list(_VICTIM_LOCATIONS)}"
        )
    if hasattr(BWR, 'value'): BWR = BWR.value   # accept Quantity or plain float
    Hm, d, freq, E_limit, BWR = (np.asarray(x) for x in (Hm, d, freq, E_limit, BWR))
    E_limit = E_limit + BWR                                    # eq. 22
    Easy40_dB, _, _ = _findE40(freq, ground_type)
    lam_r    = C_M_S / (freq * 1e6) / (2 * np.pi)
    Hm_abs   = np.power(10.0, (Hm - 120.0) / 20.0)               # dB(uA/m) -> A/m

    m1 = Hm_abs * (2 * np.pi * lam_r    * d**3) / np.sqrt(lam_r**2 + d**2)                    # eq. 1
    m2 = Hm_abs * (4 * np.pi * lam_r**2 * d**3) / np.sqrt(lam_r**4 - lam_r**2*d**2 + d**4)   # eq. 2

    crossover = 2.354 * lam_r
    coaxial   = d < crossover                                       # spec Sec. 2
    m         = np.where(coaxial, m1, m2)
    m_type    = np.where(coaxial, 'coaxial', 'coplanar')   # already dtype <U8

    ERP_kW   = (20.0 / lam_r**4) * m**2 / 1000.0                  # eq. 3 [kW]
    ERP_dBkW = 10.0 * np.log10(ERP_kW)                            # dB(kW)

    d_transition = (1000.0 * np.power(10.0, -(Easy20_dB - Easy40_dB) / 20.0)  # [m]
                    if loc == 'ground_wave' else np.nan)

    distance_m, regime = _protection_distance(
        m, ERP_dBkW, Easy40_dB, E_limit, lam_r, d_transition, loc)

    return m, m_type, d_transition, ERP_dBkW + 30.0, distance_m * 1e-3, regime


if __name__ == "__main__":
    print("This not a standalone python program! Use as module.")