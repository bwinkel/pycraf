#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu
from astropy.utils.data import get_pkg_data_filename
from astropy.units import Quantity
from ... import conversions as cnv
from ... import atm
from ...utils import check_astro_quantities
# from astropy.utils.misc import NumpyRNGContext


HEIGHTS_PROFILE = [1, 10, 30, 50]

PROFILE_LOWLAT = (
    [294.074786, 237.4778, 226.929, 270.],
    [9.0662840000e+02, 2.8485260000e+02, 1.5058940282e+01, 7.9610185204e-01],
    [1.4121645413e+01, 5.1420983832e-02, 2.8760293829e-05, 1.2778908988e-06],
    [1.9163912565e+01, 5.6351371086e-02, 3.0117880564e-05, 1.5922037041e-06],
    [1.000321953, 1.0000934535, 1.0000051497, 1.0000002288],
    [7.71111916e+01, 1.91041823e+01, 3.08083540e-02, 3.28223224e-05],
    [6.31614766e+01, 2.70773340e+01, 4.83024177e-02, 3.38613965e-05]
    )
PROFILE_MIDLAT_SUMMER = (
    [289.69681, 235.7158, 239.1281161835, 275.],
    [9.0512630000e+02, 2.8370960000e+02, 1.4998514754e+01, 7.9290741247e-01],
    [9.2511658582e+00, 6.1239834071e-02, 2.7183571711e-05, 1.2496220820e-06],
    [1.2367481486e+01, 6.6613735486e-02, 2.9997029508e-05, 1.5858148249e-06],
    [1.0002974576, 1.0000938475, 1.0000048674, 1.0000002238],
    [6.54462445e+01, 2.69454086e+01, 8.65103422e-03, 2.26897384e-05],
    [5.58605089e+01, 3.88463586e+01, 1.20693332e-02, 2.23005192e-05]
    )

PROFILE_MIDLAT_WINTER = (
    [268.8965, 218., 218., 265.],
    [8.9939800000e+02, 2.5897870000e+02, 1.3691097703e+01, 7.2378985731e-01],
    [2.5601686537e+00, 5.1486866321e-04, 2.7218907085e-05, 1.1837378270e-06],
    [3.1768361347e+00, 5.1795740000e-04, 2.7382195406e-05, 1.4475797146e-06],
    [1.000275954, 1.000092191, 1.0000048737, 1.000000212],
    [7.09268917e+01, 1.47875336e+00, 7.82710701e-02, 4.36664921e-05],
    [7.39206835e+01, 2.51660327e+00, 1.33230193e-01, 4.72982065e-05]
    )

PROFILE_HIGHLAT_SUMMER = (
    [281.9167, 225., 238.4880972095, 277.],
    [8.9871920000e+02, 2.6961380000e+02, 1.4253330015e+01, 7.5351267819e-01],
    [6.2160419038e+00, 1.9974283742e-02, 2.5902312529e-05, 1.1789617138e-06],
    [8.0867836667e+00, 2.0739334758e-02, 2.8506660030e-05, 1.5070253564e-06],
    [1.000285359, 1.0000931397, 1.000004638, 1.0000002111],
    [7.13028864e+01, 2.62542848e+01, 8.75436729e-03, 1.87094474e-05],
    [6.55326136e+01, 4.19043072e+01, 1.22895220e-02, 1.80369481e-05]
    )

PROFILE_HIGHLAT_WINTER = (
    [258.31873, 217.5, 217.5, 260.],
    [8.9319570000e+02, 2.4387180000e+02, 1.2892460426e+01, 6.8156931564e-01],
    [1.2069272815e+00, 4.8594960055e-04, 2.5690079763e-05, 1.1361236208e-06],
    [1.4387259924e+00, 4.8774360000e-04, 2.5784920851e-05, 1.3631386313e-06],
    [1.0002763674, 1.0000870128, 1.0000046, 1.0000002034],
    [7.38538040e+01, 1.47919817e+00, 7.82896205e-02, 6.11697577e-05],
    [8.53537799e+01, 2.52871044e+00, 1.33861221e-01, 6.95785906e-05]
    )


def test_elevation_from_airmass():

    args_list = [
        (1, None, cnv.dimless),
        ]
    check_astro_quantities(atm.elevation_from_airmass, args_list)

    elevation = np.array([0.1, 1, 10, 50, 89.9]) * apu.deg
    airmass = Quantity([
        36.25962353, 26.50201093, 5.60010213, 1.30540729, 1.00000152
        ], cnv.dimless)

    assert_quantity_allclose(
        atm.elevation_from_airmass(airmass),
        elevation,
        rtol=1.e-5
        )


def test_airmass_from_elevation():

    args_list = [
        (0, 90, apu.deg),
        ]
    check_astro_quantities(atm.airmass_from_elevation, args_list)

    elevation = np.array([0.1, 1, 10, 50, 89.9]) * apu.deg
    airmass = Quantity([
        36.25962353, 26.50201093, 5.60010213, 1.30540729, 1.00000152
        ], cnv.dimless)

    assert_quantity_allclose(
        atm.airmass_from_elevation(elevation),
        airmass
        )


def test_opacity_from_atten():

    args_list = [
        (1.000000000001, None, cnv.dimless),
        ]
    check_astro_quantities(atm.opacity_from_atten, args_list)

    # atten_dB = Quantity([0.1, 1, 10, 50, 100], cnv.dB)  # astropy.bug
    atten_dB = np.array([0.1, 1, 10, 50, 100]) * cnv.dB
    opacity = Quantity([
        2.30258509e-02, 2.30258509e-01, 2.30258509e+00, 1.15129255e+01,
        2.30258509e+01
        ], cnv.dimless)

    assert_quantity_allclose(
        atm.opacity_from_atten(atten_dB.to(cnv.dimless)),
        opacity
        )


def test_opacity_from_atten_zenith():

    args_list = [
        (1.000000000001, None, cnv.dimless),
        (0, 90, apu.deg),
        ]
    check_astro_quantities(atm.opacity_from_atten, args_list)

    elev = 50 * apu.deg
    # atten_dB = Quantity([0.1, 1, 10, 50, 100], cnv.dB)  # astropy.bug
    atten_dB = np.array([0.1, 1, 10, 50, 100]) * cnv.dB
    opacity = Quantity([
        1.76388252e-02, 0.17638825, 1.76388252, 8.81941258, 17.63882515
        ], cnv.dimless)

    assert_quantity_allclose(
        atm.opacity_from_atten(atten_dB.to(cnv.dimless), elev),
        opacity
        )


def test_atten_from_opacity():

    args_list = [
        (0.000000000001, None, cnv.dimless),
        (0, 90, apu.deg),
        ]
    check_astro_quantities(atm.atten_from_opacity, args_list)

    # atten_dB = Quantity([0.1, 1, 10, 50, 100], cnv.dB)  # astropy.bug
    atten_dB = np.array([0.1, 1, 10, 50, 100]) * cnv.dB
    opacity = Quantity([
        2.30258509e-02, 2.30258509e-01, 2.30258509e+00, 1.15129255e+01,
        2.30258509e+01
        ], cnv.dimless)

    assert_quantity_allclose(
        atm.atten_from_opacity(opacity),
        atten_dB
        )


def test_atten_from_opacity_zenith():

    args_list = [
        (0.000000000001, None, cnv.dimless),
        (0, 90, apu.deg),
        ]
    check_astro_quantities(atm.atten_from_opacity, args_list)

    elev = 50 * apu.deg
    # atten_dB = Quantity([0.1, 1, 10, 50, 100], cnv.dB)  # astropy.bug
    atten_dB = np.array([0.1, 1, 10, 50, 100]) * cnv.dB
    opacity = Quantity([
        1.76388252e-02, 0.17638825, 1.76388252, 8.81941258, 17.63882515
        ], cnv.dimless)

    assert_quantity_allclose(
        atm.atten_from_opacity(opacity, elev),
        atten_dB
        )


def test_refractive_index():

    args_list = [
        (1.e-30, None, apu.K),
        (1.e-30, None, apu.hPa),
        (1.e-30, None, apu.hPa),
        ]
    check_astro_quantities(atm.refractive_index, args_list)

    temp = Quantity([100, 200, 300], apu.K)
    press = Quantity([900, 1000, 1100], apu.hPa)
    press_w = Quantity([200, 500, 1000], apu.hPa)

    refr_index = Quantity(
        [1.0081872, 1.0050615, 1.00443253], cnv.dimless
        )

    assert_quantity_allclose(
        atm.refractive_index(temp, press, press_w),
        refr_index
        )
    assert_quantity_allclose(
        atm.refractive_index(
            temp.to(apu.mK), press.to(apu.Pa), press_w.to(apu.Pa)
            ),
        refr_index
        )


def test_saturation_water_pressure():

    args_list = [
        (1.e-30, None, apu.K),
        (1.e-30, None, apu.hPa),
        ]
    check_astro_quantities(atm.saturation_water_pressure, args_list)

    temp = Quantity([100, 200, 300], apu.K)
    press = Quantity([900, 1000, 1100], apu.hPa)

    press_w = Quantity(
        [2.57439748e-17, 3.23857740e-03, 3.55188758e+01], apu.hPa
        )

    assert_quantity_allclose(
        atm.saturation_water_pressure(temp, press),
        press_w
        )
    assert_quantity_allclose(
        atm.saturation_water_pressure(
            temp.to(apu.mK), press.to(apu.Pa)
            ),
        press_w
        )


def test_pressure_water_from_humidity():

    args_list = [
        (1.e-30, None, apu.K),
        (1.e-30, None, apu.hPa),
        (0, 100, apu.percent),
        ]
    check_astro_quantities(atm.pressure_water_from_humidity, args_list)

    temp = Quantity([280., 290., 295.], apu.K)
    press = Quantity([990., 980., 985.], apu.hPa)
    humid = Quantity(
        [97.625776, 34.827369, 2.59509], apu.percent
        )
    press_w = Quantity(
        [9.71865696, 6.71113205, 0.68276301], apu.hPa
        )

    assert_quantity_allclose(
        atm.pressure_water_from_humidity(temp, press, humid),
        press_w
        )


def test_humidity_from_pressure_water():

    args_list = [
        (1.e-30, None, apu.K),
        (1.e-30, None, apu.hPa),
        (1.e-30, None, apu.hPa),
        ]
    check_astro_quantities(atm.humidity_from_pressure_water, args_list)

    temp = Quantity([280., 290., 295.], apu.K)
    press = Quantity([990., 980., 985.], apu.hPa)
    humid = Quantity(
        [97.625776, 34.827369, 2.59509], apu.percent
        )
    press_w = Quantity(
        [9.71865696, 6.71113205, 0.68276301], apu.hPa
        )

    assert_quantity_allclose(
        atm.humidity_from_pressure_water(temp, press, press_w),
        humid
        )


def test_pressure_water_from_rho_water():

    args_list = [
        (1.e-30, None, apu.K),
        (1.e-30, None, apu.g / apu.m ** 3),
        ]
    check_astro_quantities(atm.pressure_water_from_rho_water, args_list)

    temp = Quantity([280., 290., 295.], apu.K)
    rho_w = Quantity([7.5, 5., 0.5], apu.g / apu.m ** 3)
    press_w = Quantity(
        [9.6908168, 6.69127826, 0.68066451], apu.hPa
        )

    assert_quantity_allclose(
        atm.pressure_water_from_rho_water(temp, rho_w),
        press_w
        )


def test_rho_water_from_pressure_water():

    args_list = [
        (1.e-30, None, apu.K),
        (1.e-30, None, apu.hPa),
        ]
    check_astro_quantities(atm.rho_water_from_pressure_water, args_list)

    temp = Quantity([280., 290., 295.], apu.K)
    rho_w = Quantity([7.5, 5., 0.5], apu.g / apu.m ** 3)
    press_w = Quantity(
        [9.6908168, 6.69127826, 0.68066451], apu.hPa
        )

    assert_quantity_allclose(
        atm.rho_water_from_pressure_water(temp, press_w),
        rho_w
        )


def test_profile_standard():

    args_list = [
        (0, 84.99999999, apu.km),
        ]
    check_astro_quantities(atm.profile_standard, args_list)

    # also testing multi-dim arrays:
    heights = Quantity([[1, 10], [3, 20], [30, 50]], apu.km)
    atm_hprof = atm.profile_standard(heights)

    assert_quantity_allclose(
        atm_hprof.temperature,
        Quantity([
            [281.65, 223.15],
            [268.65, 216.65],
            [226.65, 270.65],
            ], apu.K)
        )
    assert_quantity_allclose(
        atm_hprof.pressure,
        Quantity([
            [8.98746319e+02, 2.64364701e+02],
            [7.01086918e+02, 5.47497974e+01],
            [1.17189629e+01, 7.59478828e-01]
            ], apu.hPa)
        )
    assert_quantity_allclose(
        atm_hprof.rho_water,
        Quantity([
            [4.54897995e+00, 5.05346025e-02],
            [1.67347620e+00, 3.40499473e-04],
            [2.24089942e-05, 1.21617633e-06]
            ], apu.g / apu.m ** 3)
        )
    assert_quantity_allclose(
        atm_hprof.pressure_water,
        Quantity([
            [5.91241441e+00, 5.20387473e-02],
            [2.07466258e+00, 3.40420909e-04],
            [2.34379258e-05, 1.51895766e-06]
            ], apu.hPa)
        )
    assert_quantity_allclose(
        atm_hprof.ref_index,
        Quantity([
            [1.00027544, 1.00009232],
            [1.00021324, 1.00001961],
            [1.00000401, 1.00000022]
            ], cnv.dimless)
        )
    assert_quantity_allclose(
        atm_hprof.humidity_water,
        Quantity([
            [5.30812596e+01, 8.12224381e+01],
            [4.72174462e+01, 1.14582635e+00],
            [2.47241153e-02, 2.98350282e-05]
            ], apu.percent)
        )
    assert_quantity_allclose(
        atm_hprof.humidity_ice,
        Quantity([
            [4.89102920e+01, 1.31884489e+02],
            [4.93347383e+01, 1.97403148e+00],
            [3.88650989e-02, 3.05857185e-05]
            ], apu.percent)
        )


def test_special_profiles():

    for _profile_name in [
            'profile_lowlat',
            'profile_midlat_summer', 'profile_midlat_winter',
            'profile_highlat_summer', 'profile_highlat_winter'
            ]:

        _prof_func = getattr(atm, _profile_name)
        heights = Quantity(HEIGHTS_PROFILE, apu.km)
        consts = globals()[_profile_name.upper()]

        print(_profile_name, consts)
        (
            c_temperatures,
            c_pressures,
            c_rho_water,
            c_pressures_water,
            c_ref_indices,
            c_humidities_water,
            c_humidities_ice,
            ) = consts

        with pytest.raises(TypeError):
            _prof_func(50)

        with pytest.raises(apu.UnitsError):
            _prof_func(50 * apu.Hz)

        with pytest.raises(ValueError):
            _prof_func(-1 * apu.km)

        with pytest.raises(ValueError):
            _prof_func([-1, 10] * apu.km)

        with pytest.raises(ValueError):
            _prof_func(101 * apu.km)

        with pytest.raises(ValueError):
            _prof_func([10, 101] * apu.km)

        atm_hprof = _prof_func(heights)

        assert_quantity_allclose(
            atm_hprof.temperature,
            Quantity(c_temperatures, apu.K)
            )
        assert_quantity_allclose(
            atm_hprof.pressure,
            Quantity(c_pressures, apu.hPa)
            )
        assert_quantity_allclose(
            atm_hprof.rho_water,
            Quantity(c_rho_water, apu.g / apu.m ** 3)
            )
        assert_quantity_allclose(
            atm_hprof.pressure_water,
            Quantity(c_pressures_water, apu.hPa)
            )
        assert_quantity_allclose(
            atm_hprof.ref_index,
            Quantity(c_ref_indices, cnv.dimless)
            )
        assert_quantity_allclose(
            atm_hprof.humidity_water,
            Quantity(c_humidities_water, apu.percent)
            )
        assert_quantity_allclose(
            atm_hprof.humidity_ice,
            Quantity(c_humidities_ice, apu.percent)
            )


def test_atten_specific_annex1():

    args_list = [
        (1.e-30, None, apu.GHz),
        (1.e-30, None, apu.hPa),
        (1.e-30, None, apu.hPa),
        (1.e-30, None, apu.K),
        ]
    check_astro_quantities(atm.atten_specific_annex1, args_list)

    # test for scalar quantities
    with pytest.raises(TypeError):
        atm.atten_specific_annex1(
            1 * apu.GHz,
            Quantity([1000, 1000], apu.hPa),
            10 * apu.hPa, 300 * apu.K
            )

    with pytest.raises(TypeError):
        atm.atten_specific_annex1(
            1 * apu.GHz, 1000 * apu.hPa,
            Quantity([10, 10], apu.hPa),
            300 * apu.K
            )

    with pytest.raises(TypeError):
        atm.atten_specific_annex1(
            1 * apu.GHz, 1000 * apu.hPa, 10 * apu.hPa,
            Quantity([300, 300], apu.K)
            )
    atten_dry, atten_wet = atm.atten_specific_annex1(
        np.logspace(1, 2, 5) * apu.GHz,
        980 * apu.hPa,
        10 * apu.hPa,
        300 * apu.K
        )

    assert_quantity_allclose(
        atten_dry,
        Quantity([
            6.8734000906e-03, 8.9691079865e-03, 2.0108798019e-02,
            7.0886935962e+00, 2.6981301299e-02
            ], cnv.dB / apu.km)
        )

    assert_quantity_allclose(
        atten_wet,
        Quantity([
            0.0057314491, 0.0425094528, 0.0665904646, 0.1297056177,
            0.3999561273
            ], cnv.dB / apu.km)
        )


def test_atten_terrestrial():

    args_list = [
        (1.e-30, None, cnv.dB / apu.km),
        (1.e-30, None, apu.km),
        ]
    check_astro_quantities(atm.atten_terrestrial, args_list)

    assert_quantity_allclose(
        atm.atten_terrestrial(
            Quantity([
                0.0057314491, 0.0425094528, 0.0665904646, 0.1297056177
                ], cnv.dB / apu.km),
            Quantity(10, apu.km),
            ),
        np.array([
            0.057314491, 0.425094528, 0.665904646, 1.297056177
            ]) * cnv.dB
        )


def produce_atm_layers_test_cases():

    atm_layers_cache = atm.atm_layers(
        [1, 22, 60, 200] * apu.GHz, atm.profile_standard
        )

    np.savez('/tmp/atm_layers.npz', **atm_layers_cache)


# produce_atm_layers_test_cases()


def test_atm_layers():
    '''
    Test if the atm_layers dict is correct.
    '''

    atm_layers_cache_act = atm.atm_layers(
        [1, 22, 60, 200] * apu.GHz, atm.profile_standard
        )
    atm_layers_cache = np.load(get_pkg_data_filename('data/atm_layers.npz'))

    for k in atm_layers_cache:
        assert_quantity_allclose(atm_layers_cache_act[k], atm_layers_cache[k])


def test_atten_slant_annex1_space():

    freq_grid = np.logspace(1, 2, 5) * apu.GHz
    atm_layers_cache = atm.atm_layers(freq_grid, atm.profile_standard)

    atten, refract, tebb = atm.atten_slant_annex1(
        30 * apu.deg, 400 * apu.m, atm_layers_cache
        )

    print(atten, refract, tebb)
    assert_quantity_allclose(
        atten,
        np.array([
            9.27244289e-02, 2.29522534e-01, 4.21694382e-01, 1.50756050e+02,
            1.54722670e+00,
            ]) * cnv.dB
        )

    assert_quantity_allclose(
        refract, Quantity(-0.02961141, apu.deg), rtol=1e-6
        )

    assert_quantity_allclose(
        tebb,
        Quantity([
            8.20556359, 16.38404851, 27.16555356, 283.66148097, 83.56032149,
            ], apu.K)
        )


def test_atten_slant_annex1_nonspace():

    freq_grid = np.logspace(1, 2, 5) * apu.GHz
    atm_layers_cache = atm.atm_layers(freq_grid, atm.profile_standard)

    atten, refract, tebb = atm.atten_slant_annex1(
        5 * apu.deg, 10 * apu.m, atm_layers_cache,
        max_path_length=10 * apu.km
        )

    print(atten, refract, tebb)
    assert_quantity_allclose(
        atten,
        np.array([
            0.12751887, 0.46983218, 0.81787957, 76.38974978, 3.98743295,
            ]) * cnv.dB
        )

    assert_quantity_allclose(
        refract, Quantity(-0.024698246, apu.deg)
        )

    # Tebb only reasonable for paths into space!
    assert_quantity_allclose(
        tebb,
        Quantity([
            np.nan, np.nan, np.nan, np.nan, np.nan,
            ], apu.K)
        )


PATH_CASES_A = [
    # elev, obs_alt, max_plen, actual_plen, a_n, delta_n, h_n, refraction
    # first check vertical paths
    (90, 0, 1000, 1000, 919.38358975, 0., 1000, -0.),
    (90, 10, 1000, 1000, 919.39358975, 0., 1000.01, -0.),
    (90, 100, 10, 10., 0.03728178, 0., 10.1, -0.),
    (89.99991, 100, 10, 10., 0.03728178, 0., 10.1, -0.00000008),
    (-90, 10100, 10, 10., 0.00083376, 0., 0.1, -0.),
    (-89.99991, 10100, 10, 10., 0.00083376, 0., 0.1, 0.00000001),
    # there is a potential problem, if obs_alt is on a layer edge
    (90, 0.1, 1000, 1000, 919.38368975, 0., 1000, -0.),
    (30, 0.1, 1000, 1000, 841.55174685, 0.12542661, 553.9321417, -0.03139672),
    # other slant paths
    (45, 3000, 50, 50., 0.52294406, 0.00551708, 38.44696924, -0.01211820),
    (45, 3000, 5, 5., 0.08362677, 0.00055439, 6.53637646, -0.00419494),
    (-45, 3000, 50, 4.24343962, 0.00014147, 0.00047095, 0., -0.00600012),
    (-45, 3000, 5, 4.24343962, 0.00014147, 0.00047095, 0., -0.00600012),
    (-45, 3000, 2, 2., 0.00544606, 0.00022192, 1.58591492, -0.00242806),
    (0.01, 3000, 50, 50., 0.74177076, 0.00784422, 3.18454423, -0.06756602),
    (0.1, 3000, 50, 50., 1.56698888, 0.00784413, 3.25685427, -0.07273994),
    (-0.01, 3000, 50, 50., 3.34069529, 0.00784424, 3.16928104, -0.06108040),
    (-0.1, 3000, 50, 50., 2.33309911, 0.00784434, 3.06440499, -0.10285535),
    (-0.319 - 0.039, 3103, 50, 50., 13.80612560, 0.00784432, 2.95401998,
     -0.11288822),
    ]


# the following is very close to the above cases (~milliarcsecs), not sure
# where the (small) difference is coming from...; perhaps numerical issues?
PATH_CASES_B = [
    # elev, obs_alt, max_delta, a_n, delta_n, h_n, refraction
    (45, 3000, 0.316105, 0.52287891, 0.00551707, 38.44692292, -0.01211820),
    (45, 3000, 0.0317645, 0.08363155, 0.00055440, 6.53637984, -0.00419493),
    (-45, 3000, 0.0317645, 0.00014147, 0.00047095, 0.00000000, -0.00600012),
    # the following line results in completely different refraction
    # on all systems: win10 (-0.00020954) and macos (-0.00816826)?
    # what's going on?
    # (-45, 3000, 0.0269828, 0.00003832, 0.00047094, 0.00007292, 0.00568725),
    (-45, 3000, 0.0127149, 0.00544490, 0.00022192, 1.58591574, -0.00242803),
    (0.01, 3000, 0.4494407, 0.74176822, 0.00784422, 3.18454421, -0.06756602),
    (0.1, 3000, 0.4494353, 1.56698882, 0.00784413, 3.25685427, -0.07273994),
    (-0.01, 3000, 0.4494417, 3.34068989, 0.00784424, 3.16928101, -0.06108040),
    (-0.1, 3000, 0.44944578, 2.33289360, 0.00784431, 3.06440407, -0.10285542),
    ]


PATH_CASES_C = [
    # obs_alt, target_alt, arc_len
    (100, 100, 0.1, -0.03860530, 100),
    (6007, 23884, 0.17555, 42.33517067, 23884),
    (6430, 9523, 0.11345, 13.70828048, 9522.99999974),
    (50, 0, 0.05, -0.53275731, 0.00000003),
    (50, 0, 0.0001, -77.46204240, -0.00000000),
    (50, 0, 0.000001, -89.87258021, -0.00000000),
    (55e3, 2011.2917e3, 12.1746931, 45., 2011.2917e3),  # space path
    (70e3, 500066.13e3, 1.97456867, 88., 500066.1006e3),  # outer space path
    ]


def test_raytrace_path_pathlength():
    '''
    Test max_path_len functionality.
    '''

    freq_grid = [1] * apu.GHz  # frequency not important here
    atm_layers_cache = atm.atm_layers(freq_grid, atm.profile_standard)
    assert atm_layers_cache['space_i'] == 900

    # first test some basic properties
    path_params, refraction, is_space_path = atm.atm.raytrace_path(
        90 * apu.deg, 0 * apu.m, atm_layers_cache,
        max_path_length=1000. * apu.km
        )

    # sum over a_n (path lengths per layer) must be smaller than atm params
    # max height
    print(np.sum(path_params.a_n))
    assert_quantity_allclose(np.sum(path_params.a_n), 1000.)
    assert_quantity_allclose(refraction.to(apu.deg).value, 0.0, atol=1.e-6)

    for p in PATH_CASES_A:
        elev, obsalt_m, max_plen = p[:3]
        desired_p = p[3:]
        pp, refraction, is_space_path = atm.raytrace_path(
            elev * apu.deg, obsalt_m * apu.m, atm_layers_cache,
            max_path_length=max_plen * apu.km
            )

        # elev, obs_alt, max_plen, actual_plen, a_n, delta_n, h_n, refraction
        actual_p = (
            np.sum(pp.a_n), pp.a_n[-1], pp.delta_n[-1], pp.h_n[-1],
            refraction.to(apu.deg).value
            )
        print('{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}'.format(*actual_p))
        assert_quantity_allclose(actual_p, desired_p, atol=1.e-6)


def test_raytrace_path_arclength():
    '''
    Test max_arc_len functionality.
    '''

    freq_grid = [1] * apu.GHz  # frequency not important here
    atm_layers_cache = atm.atm_layers(freq_grid, atm.profile_standard)

    for p in PATH_CASES_B:
        elev, obsalt_m, max_alen = p[:3]
        desired_p = p[3:]
        pp, refraction, is_space_path = atm.raytrace_path(
            elev * apu.deg, obsalt_m * apu.m, atm_layers_cache,
            max_arc_length=max_alen * apu.deg
            )

        # elev, obs_alt, max_arc_len, a_n, delta_n, h_n, refraction
        actual_p = (
            pp.a_n[-1], pp.delta_n[-1], pp.h_n[-1],
            refraction.to(apu.deg).value
            )
        print('{:.8f}, {:.8f}, {:.8f}, {:.8f}'.format(*actual_p))
        assert_quantity_allclose(actual_p, desired_p, atol=1.e-6)


def test_path_endpoint_pathlength():
    '''
    Test max_path_len functionality.
    '''

    freq_grid = [1] * apu.GHz  # frequency not important here
    atm_layers_cache = atm.atm_layers(freq_grid, atm.profile_standard)

    for p in PATH_CASES_A:
        elev, obsalt_m, max_plen = p[:3]
        desired_p = p[3:]
        pp = atm.path_endpoint(
            elev * apu.deg, obsalt_m * apu.m, atm_layers_cache,
            max_path_length=max_plen * apu.km
            )

        actual_p = (
            pp.path_length.to(apu.km).value,
            pp.a_n.to(apu.km).value,
            pp.delta_n.to(apu.rad).value,
            pp.h_n.to(apu.km).value,
            pp.refraction.to(apu.deg).value
            )
        print('{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}'.format(*actual_p))
        assert_quantity_allclose(actual_p, desired_p, atol=1.e-6)


def test_path_endpoint_arclength():
    '''
    Test max_arc_len functionality.
    '''

    freq_grid = [1] * apu.GHz  # frequency not important here
    atm_layers_cache = atm.atm_layers(freq_grid, atm.profile_standard)

    for p in PATH_CASES_B:
        elev, obsalt_m, max_alen = p[:3]
        desired_p = p[3:]
        pp = atm.path_endpoint(
            elev * apu.deg, obsalt_m * apu.m, atm_layers_cache,
            max_arc_length=max_alen * apu.deg
            )

        # elev, obs_alt, max_arc_len, a_n, delta_n, h_n, refraction
        actual_p = (
            pp.a_n.to(apu.km).value,
            pp.delta_n.to(apu.rad).value,
            pp.h_n.to(apu.km).value,
            pp.refraction.to(apu.deg).value
            )
        print('{:.8f}, {:.8f}, {:.8f}, {:.8f}'.format(*actual_p))
        assert_quantity_allclose(actual_p, desired_p, atol=1.e-6)


def test_find_elevation():

    freq_grid = [1] * apu.GHz  # frequency not important here
    atm_layers_cache = atm.atm_layers(freq_grid, atm.profile_standard)

    for p in PATH_CASES_C:
        obs_alt, target_alt, arc_len = p[:3]
        desired_p = p[3:]
        elev_opt, h_opt = atm.find_elevation(
            obs_alt * apu.m, target_alt * apu.m, arc_len * apu.deg,
            atm_layers_cache,
            niter=50, interval=10, stepsize=0.05,
            seed=0,
            )

        # elev, obs_alt, max_arc_len, a_n, delta_n, h_n, refraction
        actual_p = (elev_opt.to(apu.deg).value, h_opt.to(apu.m).value)
        print('{:.8f}, {:.8f}'.format(*actual_p))
        assert_quantity_allclose(actual_p, desired_p, rtol=1.e-6, atol=1.e-6)


def test_atten_specific_annex2():

    args_list = [
        (1.e-30, None, apu.GHz),
        (1.e-30, None, apu.hPa),
        (1.e-30, None, apu.g / apu.m ** 3),
        (1.e-30, None, apu.K),
        ]
    check_astro_quantities(atm.atten_specific_annex2, args_list)

    # test for scalar quantities
    with pytest.raises(TypeError):
        atm.atten_specific_annex2(
            1 * apu.GHz,
            Quantity([1000, 1000], apu.hPa),
            10 * apu.g / apu.m ** 3, 300 * apu.K
            )

    with pytest.raises(TypeError):
        atm.atten_specific_annex2(
            1 * apu.GHz, 1000 * apu.hPa,
            Quantity([10, 10], apu.g / apu.m ** 3),
            300 * apu.K
            )

    with pytest.raises(TypeError):
        atm.atten_specific_annex2(
            1 * apu.GHz, 1000 * apu.hPa, 10 * apu.g / apu.m ** 3,
            Quantity([300, 300], apu.K)
            )
    atten_dry, atten_wet = atm.atten_specific_annex2(
        np.logspace(1, 2, 5) * apu.GHz,
        980 * apu.hPa,
        10 * apu.g / apu.m ** 3,
        300 * apu.K
        )

    assert_quantity_allclose(
        atten_dry,
        Quantity([
            0.006643591, 0.008556794, 0.0200009175, 6.5824558517,
            0.0215553521
            ], cnv.dB / apu.km)
        )

    assert_quantity_allclose(
        atten_wet,
        Quantity([
            0.0082768112, 0.0599180804, 0.096043373, 0.1902800424,
            0.5888230557
            ], cnv.dB / apu.km)
        )


def test_equivalent_height_dry():

    args_list = [
        (1.e-30, None, apu.GHz),
        (1.e-30, None, apu.hPa),
        ]
    check_astro_quantities(atm.equivalent_height_dry, args_list)

    height = atm.equivalent_height_dry(
        np.logspace(1, 2, 5) * apu.GHz,
        980 * apu.hPa,
        )
    print(height)

    assert_quantity_allclose(
        height,
        Quantity([
            5.17174463, 5.15765153, 5.12390758, 10.2730745, 5.38128124
            ], apu.km)
        )


def test_equivalent_height_wet():

    args_list = [
        (1.e-30, None, apu.GHz),
        (1.e-30, None, apu.hPa),
        ]
    check_astro_quantities(atm.equivalent_height_wet, args_list)

    height = atm.equivalent_height_wet(
        np.logspace(1, 2, 5) * apu.GHz,
        980 * apu.hPa,
        )
    print(height)

    assert_quantity_allclose(
        height,
        Quantity([
            1.67507787, 1.7615659, 1.68523706, 1.66232892, 1.66121491
            ], apu.km)
        )


def test_atten_slant_annex2():

    args_list = [
        (1.e-30, None, cnv.dB / apu.km),
        (1.e-30, None, cnv.dB / apu.km),
        (1.e-30, None, apu.km),
        (1.e-30, None, apu.km),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(atm.atten_slant_annex2, args_list)

    atten_dry = Quantity([
        0.006643591, 0.008556794, 0.0200009175, 6.5824558517,
        0.0215553521
        ], cnv.dB / apu.km)

    atten_wet = Quantity([
        0.0082768112, 0.0599180804, 0.096043373, 0.1902800424,
        0.5888230557
        ], cnv.dB / apu.km)

    atten_tot = atm.atten_slant_annex2(
        atten_dry, atten_wet,
        10 * apu.km, 1 * apu.km, 30 * apu.deg
        )

    assert_quantity_allclose(
        atten_tot,
        np.array([
            0.1494254424, 0.2909720408, 0.592105096,
            132.0296771188, 1.6087531534
            ]) * cnv.dB
        )
