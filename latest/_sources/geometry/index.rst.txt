.. pycraf-geometry:

**************************************
Geometry helpers (`pycraf.geometry`)
**************************************

.. currentmodule:: pycraf.geometry

Introduction
============

The `~pycraf.geometry` sub-package just offers some convenience functions
for various geometrical calculations, e.g., in spherical coordinates.


Using `pycraf.geometry`
=======================

Spherical coordinates
---------------------

One can convert between cartesian and spherical coordinates using the two
functions `~pycraf.geometry.cart_to_sphere` and
`~pycraf.geometry.sphere_to_cart`::

    >>> from pycraf import geometry
    >>> from astropy import units as u
    >>> import numpy as np

    >>> r, az, el = 1 * u.m, 0 * u.deg, 90 * u.deg  # z-axis
    >>> geometry.sphere_to_cart(r, az, el)  # doctest: +FLOAT_CMP
    (<Quantity 0.0 m>, <Quantity 0.0 m>, <Quantity 1.0 m>)

    >>> x, y, z = 200 * u.m, 1.1 * u.km, 500 * u.m
    >>> geometry.cart_to_sphere(x, y, z)  # doctest: +FLOAT_CMP
    (<Quantity 1224.745 m>, <Quantity 79.69515 deg>, <Quantity 24.09484 deg>)

.. note::

    In contrast to the often-used mathematical convention, pycraf does not
    work with the zenith distances but height above the horizon (i.e.,
    elevation).

Very useful are the two functions `~pycraf.geometry.true_angular_distance`
and `~pycraf.geometry.great_circle_bearing`, which allow to determine
the true angular separation between two points on the sphere and the
bearing of a point on the sphere w.r.t. another point::

    >>> l1, b1 = 25 * u.deg, 34 * u.deg
    >>> l2, b2 = 19 * u.deg, 54 * u.deg
    >>> geometry.true_angular_distance(l1, b1, l2, b2)  # doctest: +FLOAT_CMP
    <Quantity 20.4425 deg>
    >>> geometry.great_circle_bearing(l1, b1, l2, b2)  # doctest: +FLOAT_CMP
    <Quantity -10.1317 deg>

.. warning::

    If you need angular distances/bearings on the Geoid with high accuracy
    use the Geodesics methods in the `~pycraf.pathprof` subpackage.



Rotation matrices
-----------------

Sometimes one needs to rotate quantities (e.g., antenna diagrams) in 3D space
when using compatibility studies. In pycraf a few routines are provided that
offer basic functionality. Rotations are easily performed using matrix
algebra, where :math:`\vec y=R\vec x`. Rotation matrices must be orthogonal
(:math:`R^{-1}=R^\mathrm{T}`) and have determinant :math:`\det R=+1`.

Two easy ways are implemented in pycraf to construct rotation matrices. The
first is via concatenation, :math:`R=R_z(\alpha_3)R_y(\alpha_2)R_x(\alpha_1)`, of the three basic rotation matrices:

.. math::

    R_x = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos \theta & -\sin \theta \\
    0 & \sin \theta & \cos \theta
    \end{bmatrix},
    R_y = \begin{bmatrix}
    \cos \theta & 0 & \sin \theta\\
    0 & 1 & 0 \\
    -\sin \theta & 0 & \cos \theta
    \end{bmatrix},
    R_z = \begin{bmatrix}
    \cos \theta & -\sin \theta & 0 \\
    \sin \theta & \cos \theta & 0 \\
    0 & 0 & 1
    \end{bmatrix}

.. note::
    One can also use other combinations of the basic rotation matrices (in
    total there are 24 different possibilities). Apart from the 'xyz' order,
    a common scheme is 'zxz', i.e.,
    :math:`R=R_z(\alpha_3)R_x(\alpha_2)R_z(\alpha_1)`.
    The rotation angles are often called Euler angles.

.. note::
    The right-most matrix is applied first, but since rotation matrices are
    associative one can just calculate the matrix multiplication (in any
    order) and apply the result to the target vector.

Example::

    >>> R = geometry.multiply_matrices(
    ...     geometry.Rz(10 * u.deg),
    ...     geometry.Ry(-30 * u.deg),
    ...     geometry.Rx(-15 * u.deg),
    ...     )
    >>> R  # doctest: +FLOAT_CMP
    array([[ 0.85286853, -0.04028776, -0.52056908],
           [ 0.15038373,  0.97372297,  0.17102137],
           [ 0.5       , -0.22414387,  0.8365163 ]])

    >>> vector = (1, 4, -2) * u.m
    >>> np.matmul(R, vector)  # doctest: +FLOAT_CMP
    <Quantity [ 1.73286, 3.70323,-2.06961] m>

The second approach is by specifying a rotation axis and angle and use
`Rodrigues' Rotation Formula
<https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>`__::

    >>> rotax = (0.5, 0.2, -0.8) * u.m
    >>> rotang = -10 * u.deg
    >>> R2 = geometry.rotmat_from_rotaxis(*rotax, rotang)
    >>> R2  # doctest: +FLOAT_CMP
    array([[ 0.98889169, -0.14241824, -0.04254725],
           [ 0.14568539,  0.98546118,  0.08741867],
           [ 0.02947865, -0.09264611,  0.99526263]])

It is possible to extract the Euler angles or the rotation axis/angle from
a given rotation matrix, with the functions
`~pycraf.geometry.eulerangle_from_rotmat` and
`~pycraf.geometry.rotaxis_from_rotmat`, but keep in mind that the solution
is not unique!



See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.
- `Spherical coordinates
  <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_
- `Rotation matrix <https://en.wikipedia.org/wiki/Rotation_matrix>`__
- `Rodrigues' Rotation Formula
  <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>`__


Reference/API
=============

.. automodapi:: pycraf.geometry
    :no-inheritance-diagram:
