import numpy as np
import astropy
import astropy.units as apu
from astropy import constants
from .. import protection, utils, conversions as cnv
from .. utils import ranged_quantity_input

"""
Compute the separation distance for a given frequency, radiated emission, and the distance at which the emission is referenced,
along with the protection criteria. This calculation adheres to the methodology outlined in Recommendation ITU-R SM.2028-0 (2012),
which provides guidelines for estimating the ground-wave propagation and separation distances.
"""

# Physical constants with explicit units

# c = constants.c
# u0 = constants.mu0  # Permeability of free space
# eps0 = constants.eps0  # Permittivity of free space

# Fixed field strength at the roll-off of 20 dB/decade

# Easy20_dB = 109.5 * cnv.dB_uV_m  # Field strength in dB(µV/m)
Easy20_dB = 109.5  # Field strength in dB(µV/m)

# Characteristic impedance of free space (around 377 ohms)
# Z0 = np.sqrt(u0 / eps0).to(apu.ohm)


__all__ = [
    "findE40",
    "ERP2dist_ITU368",
]

TAB368_KHZ = np.array([
    10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 
    1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 30000
    ])

TAB368_TYPES = np.array([
    [
        166, 164, 163, 162, 162, 161, 160, 159, 158, 158, 157, 156, 156, 154, 
        152, 151, 150, 147, 144, 142, 136, 132, 126, 120, 113
    ],
    [
        166, 165, 164, 163, 162, 161, 160, 159, 158, 158, 157, 156, 155, 154, 
        153, 153, 152, 151, 149, 148, 146, 143, 138, 134, 127
    ],
    [
        165, 164, 163, 162, 161, 159, 157, 155, 151, 147, 141, 136, 132, 126, 
        122, 118, 115, 111, 108, 107, 103, 101, 97, 95, 91
    ],
    [
        167, 165, 164, 163, 162, 162, 161, 160, 158, 157, 155, 153, 150, 146, 
        142, 135, 129, 123, 117, 113, 105, 100, 95, 91, 87
    ],
    [
        165, 163, 163, 162, 161, 161, 159, 158, 156, 154, 150, 147, 143, 137, 
        132, 124, 119, 112, 107, 103, 97, 94, 89, 87, 83
    ],
    [
        165, 164, 163, 163, 162, 161, 158, 156, 153, 148, 142, 135, 134, 127, 
        120, 114, 109, 103, 99, 97, 93, 90, 87, 84, 80
    ],
    [
        165, 164, 163, 161, 160, 158, 154, 150, 144, 140, 132, 127, 123, 117, 
        112, 107, 103, 98, 95, 93, 89, 87, 83, 81, 77
    ],
    [
        164, 163, 162, 158, 155, 152, 146, 142, 134, 129, 122, 117, 113, 107, 
        103, 98, 95, 93, 90, 87, 84, 81, 77, 75, 72
    ],
    [
        163, 160, 157, 152, 148, 144, 137, 132, 124, 119, 112, 107, 103, 98, 
        96, 92, 89, 86, 83, 81, 78, 76, 72, 70, 66
    ],
    [
        159, 154, 149, 142, 137, 133, 126, 121, 115, 111, 107, 104, 102, 98, 
        96, 92, 89, 84, 81, 76, 78, 76, 72, 70, 66
    ],
    [
        151, 144, 139, 132, 128, 124, 119, 116, 112, 109, 106, 103, 102, 98, 
        96, 92, 89, 83, 82, 78, 76, 72, 70, 70, 66
    ]
    ])    



# Ground types with their respective conductivity and permittivity values
#   - Type of land or ground
#   - s: Conductivity in appropriate units (S/m, mS/m, or uS/m)
#   - e: Relative permittivity (dimensionless)
Ground_Types = {
    1: ("Sea water, low salinity", 1 * apu.S / apu.m, 80),
    2: ("Sea water, average salinity", 5 * apu.S / apu.m, 70),
    3: ("Fresh water", 3 * apu.mS / apu.m, 80),
    4: ("Land (very wet)", 30 * apu.mS / apu.m, 40),
    5: ("Wet ground", 10 * apu.mS / apu.m, 30),
    6: ("Land", 3 * apu.mS / apu.m, 22),
    7: ("Medium dry ground", 1 * apu.mS / apu.m, 15),
    8: ("Dry ground", 0.3 * apu.mS / apu.m, 7),
    9: ("Very dry ground", 0.1 * apu.mS / apu.m, 3),
    10: ("Fresh water ice, -1 °C", 30 * apu.uS / apu.m, 3),
    11: ("Fresh water ice, -10 °C", 10 * apu.uS / apu.m, 3),
}


def _findE40(freq, key_ground_term):
    # Tabulated data for field strength (Easy40_dB) based on ground types and frequencies


    # Define frequency range for ITU-R P.368 model
    #     MIN_VAL, MAX_VAL = 10 * apu.kHz, 30 * apu.MHz

    #     # Check if the frequency is within the allowed range
    #     if not (MIN_VAL <= freq <= MAX_VAL):
    #         raise ValueError(f"Frequency {freq} is out of range [{MIN_VAL}, {MAX_VAL}]. Allowed range is 10 kHz to 30 MHz.")

    # Match the ground type based on the user-provided term and find the closest matching frequency index

    freq_kHz = freq * 1e3

    ground_type_num = next(
        (
            key
            for key, value in Ground_Types.items()
            if key_ground_term.lower() == value[0].lower()
        ),
        None,
    )
    if ground_type_num is None:
        raise ValueError(
            f"Ground type '{key_ground_term}' not recognized. Please use a valid ground type."
        )
    cond_gr, permit_gr = Ground_Types[ground_type_num][1:]

    idx = np.argmin(np.abs(TAB368_KHZ - freq_kHz))

    # Retrieve the field strength for the selected ground type and frequency

    # idx_ground_type = f"type {ground_type_num:d}"
    Easy40_dB = TAB368_TYPES[ground_type_num - 1][idx] * apu.dB(apu.uV / apu.m)

    return Easy40_dB, cond_gr.to_value(apu.S / apu.m), permit_gr


@utils.ranged_quantity_input(
    freq=(0.01, 30, apu.MHz),
    #key_ground_term=(None, None, None),
    strip_input_units=True, output_unit=(cnv.dB_uV_m, apu.S / apu.m, cnv.dimless)
    )
def findE40(freq, key_ground_term):
    """
    Find the field strength (Easy40_dB) for a given ground type and frequency based on ITU-R P.368 data.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        The frequency of the signal in Hz (must be between 10 kHz and 30 MHz).
    key_ground_term : str
        The description of the ground type (e.g., "Land", "Wet ground").

    Returns
    -------
    Easy40_dB : `~astropy.units.Quantity`
        The field strength in dB(µV/m).
    cond_gr : 

    permit_gr : 
    

    Raises
    ------
    ValueError
        If the ground type or frequency is out of range.
    """

    return _findE40(freq, key_ground_term)



def _gw_prop(Hm, d, freq, key_ground_term, E_limit):

    # Compute radian wavelength

    # freq in MHz
    # lam_r in meters
    lam_r = (constants.c.to_value(apu.m / apu.s) / freq / 1e6) / (2 * np.pi)

    # Convert magnetic field strength to linear units

    # Hm = dBux_m_2_x_m(Hm_dBu)
    # Hm = Hm_dBapu.to(apu.A / apu.m) *10
    # Hm = 10*Hm_dBapu.physical    # -20 dBuA/m --> 1e-7 A/m

    # Compute magnetic dipole moments in the coaxial and coplanar directions

    # units of A * m ** 2
    m1 = (
        np.abs(Hm)
        * (2 * np.pi * lam_r * d ** 3)
        / np.sqrt(lam_r ** 2 + d ** 2)
    )
    m2 = (
        np.abs(Hm)
        * (4 * np.pi * lam_r ** 2 * d ** 3)
        / np.sqrt(lam_r**4 - (lam_r * d) ** 2 + d ** 4)
    )

    # Choose the type of dipole moment (coaxial or coplanar)
    # m_type, m = ("m1 (coaxial)", m1) if m1 > m2 else ("m2 (coplanar)", m2)
    m_mask = m1 > m2
    m = np.where(m_mask, m1, m2)
    m_type = np.where(m_mask, 'coaxial', 'coplanar')
    
    # Calculate ERP (Equivalent Radiated Power) based on the selected dipole moment
    # ERP in Watts
    ERP = (constants.mu0 * constants.c).to_value(apu.Ohm) / (6 * np.pi) * m ** 2 / lam_r ** 4
    ERP_dBkW = 10 * np.log10(ERP * 1e-3)

    # Get field strength (Easy40_dB) for the selected ground type and frequency

    Easy40_dB, _, _ = _findE40(freq, key_ground_term)

    # Compute the transition distance

    d_transition = (
        1000 * np.power(10, -(Easy20_dB - Easy40_dB) / 20)
    )  # in meters

    # Calculate the interference level
    # Eint = (Easy40_dB.value + ERP.to_value(apu.dB(apu.kW))) * apu.dB(apu.uV / apu.m)
    Eint = (Easy40_dB + ERP_dBkW)   # in dB_uV_m

    # Compute the ground-wave propagation distance

    distance_gw = (
        1000 * np.power(10, (Eint - E_limit) / 40)  # in meters
    ) * 1e-3  # in kilometers

    return m, m_type, d_transition, ERP_dBkW + 30, distance_gw


# Calculation of separation distance to meet protection criteria
@utils.ranged_quantity_input(
    Hm=(None, None,  apu.A / apu.m),
    d=(0, None, apu.m),
    freq=(0.01, 30, apu.MHz),
    #key_ground_term=(None, None, None),
    E_limit=(None, None, apu.dB(apu.uV / apu.m)),
    strip_input_units=True, output_unit=(apu.A * apu.m ** 2, None, apu.m, cnv.dB_W, apu.km),
    )
def gw_prop(Hm_dBu, d, freq, key_ground_term, E_limit):
    """
    Compute the equivalent radiated power (ERP) and ground-wave separation distance
    for a given frequency, ground type

    Parameters
    ----------
    Hm : `~astropy.units.Quantity`
        Magnetic field strength in dB(µA/m).
    d : `~astropy.units.Quantity`
        Distance at which the magnetic field strength is measured [m].
    freq : `~astropy.units.Quantity`
        Frequency of radiation [Hz].
    key_ground_term : str
        Ground type as a string (e.g., "Land").
    E_limit : `~astropy.units.Quantity`
        Electric field strength threshold limit (for RAS, it can be derived by ITU-R RA.769-2).

    Returns
    -------
    tuple
        m_type : str
            Indicates whether the coaxial or coplanar dipole moment dominates.
        d_transition : `~astropy.units.Quantity`
            Transition distance at 20 dB/decade in meters.
        ERP : `~astropy.units.Quantity`
            Equivalent Radiated Power in Watts.
        distance_gw : `~astropy.units.Quantity`
            Ground-wave propagation distance in kilometers.

    Raises
    ------
    ValueError
        If invalid data is encountered during calculations.
    """

    return _gw_prop(Hm_dBu, d, freq, key_ground_term, E_limit)

if __name__ == "__main__":
    print("This not a standalone python program! Use as module.")
