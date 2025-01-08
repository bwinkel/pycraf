import numpy as np
from astropy import units as u
import scipy.constants
from pycraf import protection

# Physical constants with explicit units using astropy
c = scipy.constants.c * u.meter / u.second  
pi = np.pi
u0 = scipy.constants.mu_0 * u.meter * u.kg / (u.second * u.A)**2  # Permeability of free space
eps0 = scipy.constants.epsilon_0 * u.C**2 / (u.N * u.m**2)  # Permittivity of free space

# Fixed field strength at the roll-off of 20 dB/decade (used in P.368-10 calculations)
Easy20_dB = 109.5 * u.dB(u.uV / u.m)  # Field strength in dB(µV/m)

# Characteristic impedance of free space (around 377 ohms)
Z0 = np.sqrt(u0 / eps0).to(u.ohm) 

# Data Lookup Function
def findE40(freq, key_ground_term):
    """
    Find the field strength (Easy40_dB) for a given ground type and frequency based on ITU-R P.368 data.
    
    Parameters:
    freq : `~astropy.units.Quantity`
        The frequency of the signal in Hz (must be between 10 kHz and 30 MHz).
    
    key_ground_term : str
        The description of the ground type (e.g., "Land", "Wet ground").
    
    Returns:
    `~astropy.units.Quantity`:
        The field strength in dB(µV/m).
    
    Raises:
    ValueError:
        If the ground type or frequency is out of range.
    """
    
    # Tabulated data for field strength (Easy40_dB) based on ground types and frequencies
    dataTAB368 = {
        "fkHz":    [10,  15,   20, 30,   40,  50, 75,  100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 30000],
        "type 1":  [166, 164, 163, 162, 162, 161, 160, 159, 158, 158, 157, 156, 156, 154, 152,   151,  150,  147,  144,  142,   136,   132,  126,  120,  113],
        "type 2":  [166, 165, 164, 163, 162, 161, 160, 159, 158, 158, 157, 156, 155, 154, 153,   153,  152,  151,  149,  148, 146, 143, 138, 134, 127],
        "type 3":  [165, 164, 163, 162, 161, 159, 157, 155, 151, 147, 141, 136, 132, 126, 122,  118,  115,   111,  108,  107, 103, 101, 97, 95, 91],
        "type 4":  [167, 165, 164, 163, 162, 162, 161, 160, 158, 157, 155, 153, 150, 146, 142,  135,  129,   123,  117,  113, 105, 100, 95, 91, 87],
        "type 5":  [165, 163, 163, 162, 161, 161, 159, 158, 156, 154, 150, 147, 143, 137, 132,  124,  119,  112,   107,  103, 97, 94, 89, 87, 83],
        "type 6":  [165, 164, 163, 163, 162, 161, 158, 156, 153, 148, 142, 135, 134, 127, 120,  114,  109,  103,    99,  97, 93, 90, 87, 84, 80],
        "type 7":  [165, 164, 163, 161, 160, 158, 154, 150, 144, 140, 132, 127, 123, 117, 112,  107,  103,  98,     95,  93, 89, 87, 83, 81, 77],
        "type 8":  [164, 163, 162, 158, 155, 152, 146, 142, 134, 129, 122, 117, 113, 107, 103,   98,   95,  93,    90,  87, 84, 81, 77, 75, 72],
        "type 9":  [163, 160, 157, 152, 148, 144, 137, 132, 124, 119, 112, 107, 103, 98, 96, 92, 89,   86,  83,    81,  78, 76, 72, 70, 66],
        "type 10": [159, 154, 149, 142, 137, 133, 126, 121, 115, 111, 107, 104, 102, 98, 96, 92, 89,   84,  81,    76,  78, 76, 72, 70, 66],
        "type 11": [151, 144, 139, 132, 128, 124, 119, 116, 112, 109, 106, 103, 102, 98, 96, 92, 89,   83,  82,    78,  76, 72, 70, 70, 66]
    }
    
    # Ground types with their respective conductivity and permittivity values
    Ground_Types = {    
        1: ("Sea water, low salinity",        "σ= 1 S/m",    "ε = 80"),
        2: ("Sea water, average salinity",    "σ= 5 S/m",    "ε = 70"),
        3: ("Fresh water",                    "σ= 3 mS/m",   "ε = 80"),
        4: ("Land (very wet)",                "σ= 30 mS/m",  "ε = 40"),
        5: ("Wet ground",                     "σ= 10 mS/m",  "ε = 30"),
        6: ("Land",                           "σ= 3 mS/m",   "ε = 22"),
        7: ("Medium dry ground",              "σ= 1 mS/m",   "ε = 15"),
        8: ("Dry ground",                     "σ= 0.3 mS/m", "ε = 7"),
        9: ("Very dry ground",                "σ= 0.1 mS/m", "ε = 3"),
        10: ("Fresh water ice, -1 °C",        "σ= 30 uS/m",  "ε = 3"),
        11: ("Fresh water ice, -10 °C",       "σ= 10 uS/m",  "ε = 3"),
    }
    # Define frequency range for ITU-R P.368 model (convert to MHz for comparison)
    MIN_VAL, MAX_VAL = 10 * u.kHz, 30 * u.MHz
    
    # Check if the frequency is within the allowed range
    if not (MIN_VAL <= freq <= MAX_VAL):
        raise ValueError(f"Frequency {freq} is out of range [{MIN_VAL}, {MAX_VAL}]. Allowed range is 10 kHz to 30 MHz.")
    
    # Match the ground type based on the user-provided term and find the closest matching frequency index
    ground_type_num = next((key for key, value in Ground_Types.items() if key_ground_term.lower() == value[0].lower()), None)
    if ground_type_num is None:
        raise ValueError(f"Ground type '{key_ground_term}' not recognized. Please use a valid ground type.")
    
    freq_arr = np.array(dataTAB368['fkHz']) * u.kHz
    idx = np.argmin(np.abs(freq_arr.to(u.kHz) - freq.to(u.kHz)))
    
    # Retrieve the field strength for the selected ground type and frequency
    idx_ground_type = f'type {ground_type_num:d}'
    Easy40_dB = dataTAB368[idx_ground_type][idx] * u.dB(u.uV / u.m)
    
    return Easy40_dB

# ERP to Distance Calculation
def ERP2dist_ITU368(Hm, d, freq, key_ground_term, E_limit):
    """
    Compute the equivalent radiated power (ERP) and ground-wave propagation distance for a given frequency, ground type, and RAS mode.
    
    Parameters:
    Hm : `~astropy.units.Quantity`
        Magnetic field strength in dB(µA/m).
    d : `~astropy.units.Quantity`
        Distance at which the Magnetic field strength is measured  [m].
    
    freq : `~astropy.units.Quantity`
        Frequency of radiation [Hz].
    
    key_ground_term : str
        Ground type as a string (e.g., "Land").
    
    E_limit : str
        Electric field strenght threshold limit (for RAS it can be derived by ITU-R RA.769-2)
    
    Returns:
    tuple:
        - m_type (str): Indicates whether the coaxial or coplanar dipole moment dominates.
        - d_transition (`~astropy.units.Quantity`): Transition distance at 20 dB/decade in meters.
        - distance_gw (`~astropy.units.Quantity`): Ground-wave propagation distance in kilometers.
    
    Raises:
    ValueError: If invalid data is encountered during calculations.
    """
         
    # Compute radian wavelength 
    lam_r = ((c / freq) / (2 * pi)).to(u.m)
    
    # Compute magnetic dipole moments in the coaxial and coplanar directions
    m1 = np.abs(Hm.to(u.A / u.m)) * (2 * pi * lam_r * d**3) / np.sqrt(lam_r**2 + d**2)
    m2 = np.abs(Hm.to(u.A / u.m)) * (4 * pi * lam_r**2 * d**3) / np.sqrt(lam_r**4 - (lam_r * d)**2 + d**4)
    
    m_type, m = ('m1 (coaxial)', m1) if m1 > m2 else ('m2 (coplanar)', m2)
    
    # Calculate ERP (Equivalent Radiated Power) based on the selected dipole moment
    ERP = u0 * c / (6 * pi) * m**2 / lam_r**4
    
    # Get field strength (Easy40_dB) for the selected ground type and frequency
    Easy40_dB = findE40(freq, key_ground_term)
    
    # Compute the transition distance (see ITU-R P.368)
    d_transition = 1000 * np.power(10, -(Easy20_dB.value - Easy40_dB.value) / 20) * u.m
        
    # Calculate the interference level
    Eint = (Easy40_dB.value + ERP.to_value(u.dB(u.kW))) * u.dB(u.uV / u.m)
    
    # Compute the ground-wave propagation distance
    distance_gw = (1000 * np.power(10, (Eint.value - E_limit.to_value(cnv.dB_uV_m)) / 40) * u.m).to(u.km)
    
    return m_type, d_transition, distance_gw
