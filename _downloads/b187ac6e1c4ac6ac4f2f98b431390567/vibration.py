"""
Analytical Hessians and Vibrational Analysis
============================================

This tutorial demonstrates how compute analytical hessians and do
vibrational analysis using ``nnp.vib``.
"""
###############################################################################
# Let's first import all the packages we will use:
import torch
import math
import pytest
from pytest import approx
import sys
import nnp.so3 as so3
import nnp.vib as vib


###############################################################################
# In this tutorial, we will study atoms moving in a quadratic potential.
# There is no interaction between these atoms.
#
# Let's first construct such a potential. A naive potential would be:
#
# .. math::
#   U(x, y, z) = \frac{1}{2} \left(0.5 x^2 + y^2 + 2 z^2\right)
def naive_potential(x, y, z):
    return 0.5 * (0.5 * x ** 2 + y ** 2 + 2 * z ** 2)


###############################################################################
# A naive potential is not very interesting, let's rotate the potential along
# the z axis for 45 degrees. Rotating the potential along the z axis for 45 degrees
# is equivalent to rotate the coordinates along the z axis for -45 degrees before
# evaluating the naive potential.  Let's make our potential able to handle both
# single molecule (i.e. coordinates has shape `(atoms, 3)`), and in batch (i.e.
# coordinates has shape `(molecules, atoms, 3)`):
rot45 = so3.rotate_along(torch.tensor([0, 0, math.pi / 4]))
rot_neg_45 = so3.rotate_along(torch.tensor([0, 0, -math.pi / 4]))


def potential(coordinates):
    rotated_coordinates = (rot_neg_45 @ coordinates.transpose(-1, -2)).transpose(-1, -2)
    x, y, z = rotated_coordinates.unbind(-1)
    return naive_potential(x, y, z).sum(dim=-1)


###############################################################################
# Analytical Hessian
# ------------------
#
# Now let's compute the hessian one molecule containing two atoms at the origin
coordinates = torch.zeros(2, 3, requires_grad=True)
energy = potential(coordinates)
hessian = vib.hessian(coordinates, energies=energy)
print(hessian)


###############################################################################
# Let's compute the theoretical result to see if it matches with the result above:
#
# First of all, because there are no interactions between atoms, the hessian
# should be a block diagonal matrix:
#
# .. math::
#   H = \left[9\times9\right] = \left[\begin{array}{cc}
#                                     3\times3\\
#                                     & 3\times3
#                                     \end{array}\right]
#
# The two :math:`3\times3` matricies are identical. For each :math:`3\times3`
# matrix, the potential is not rotated at the z axis, so the structure of it
# should be:
#
# .. math::
#   \left[\begin{array}{ccc}
#    ? & ?\\
#    ? & ?\\
#    &  & 2
#    \end{array}\right]
#
# It is not hard to figure out that, for the rotated potential, considering
# only the contribution from x,y plane, it can be written as:
#
# .. math::
#   U=\frac{1}{2}\left[0.5\left(\frac{x+y}{\sqrt{2}}\right)^{2}+\left(\frac{y-x}{\sqrt{2}}\right)^{2}\right]=\frac{1}{4}\left(1.5x^{2}+1.5y^{2}-xy\right)
#
# Therefore, the :math:`2\times2` block on the top left should be
#
# .. math::
#   \left[\begin{array}{cc}
#    0.75 & -0.25\\
#    -0.25 & 0.75
#    \end{array}\right]
def test_analytical_hessian():
    hessian00 = hessian[:3, :3]
    hessian01 = hessian[:3, 3:]
    hessian10 = hessian[3:, :3]
    hessian11 = hessian[3:, 3:]
    expected = torch.tensor([
        [ 0.75, -0.25, 0],  # noqa: E201, E241
        [-0.25,  0.75, 0],  # noqa: E201, E241
        [ 0.00,  0.00, 2],  # noqa: E201, E241
    ])
    assert torch.allclose(hessian00, expected)
    assert torch.allclose(hessian11, expected)
    assert torch.allclose(hessian10, torch.zeros(3, 3))
    assert torch.allclose(hessian01, torch.zeros(3, 3))


###############################################################################
# We also support compute multiple molecules in batch
coordinates_batch = torch.zeros(2, 2, 3, requires_grad=True)
energy_batch = potential(coordinates_batch)
hessian_batch = vib.hessian(coordinates_batch, energies=energy_batch)
print(hessian_batch)


###############################################################################
# The hessian should be just the stack of the previous results twice
def test_analytical_hessian_batch():
    expected = torch.stack([hessian, hessian])
    assert torch.allclose(expected, hessian_batch)


###############################################################################
# Vibrational Analysis
# --------------------
#
# Now let's do vibrational analysis using the computed hessians. Let's assume
# the two atoms has mass (1, 3). The hessian should have shape `(3 * atoms, 3 * atoms)`,
# the mass should have shape `(atoms,)`
mass = torch.tensor([1.0, 3.0])
freq_modes1 = vib.vibrational_analysis(mass, hessian)

###############################################################################
# The output angular frequencies should have shape `(modes,)`, while the modes
# have shape `(modes, atoms, 3)`, where the modes dimension correspond to different
# vibrational modes
print(freq_modes1.angular_frequencies.shape)
print(freq_modes1.angular_frequencies)
print(freq_modes1.modes.shape)
print(freq_modes1.modes)


###############################################################################
# The angular frequency of a harmonic oscillator is given by :math:`\sqrt{\frac{k}{m}}`.
# For this example, the two atoms are independent, therefore the normal modes would
# have one atom moving while the other static. The angular frequencies are also
# independent to each other. Therefore, we expect to see the following combinations
# of angular frequencies and normal modes:
#
# +----------------------------+-------------+-----------+
# | Angular Frequency          | Moving Atom | Direction |
# +============================+=============+===========+
# | :math:`\frac{1}{\sqrt{2}}` | 1           | (x, y)    |
# +----------------------------+-------------+-----------+
# | :math:`1`                  | 1           | (x, -y)   |
# +----------------------------+-------------+-----------+
# | :math:`\sqrt{2}`           | 1           | z         |
# +----------------------------+-------------+-----------+
# | :math:`\frac{1}{\sqrt{6}}` | 2           | (x, y)    |
# +----------------------------+-------------+-----------+
# | :math:`\frac{1}{\sqrt{3}}` | 2           | (x, -y)   |
# +----------------------------+-------------+-----------+
# | :math:`\sqrt{\frac{2}{3}}` | 2           | z         |
# +----------------------------+-------------+-----------+
#
# Now let's write a test function to test this case
def test_vibrational_analysis():
    expected_freqs = torch.tensor([
        1 / math.sqrt(6), 1 / math.sqrt(3), 1 / math.sqrt(2),
        math.sqrt(2 / 3), 1, math.sqrt(2)])
    assert torch.allclose(freq_modes1.angular_frequencies, expected_freqs)

    mode0 = freq_modes1.modes[0]
    atom0, atom1 = mode0
    assert torch.allclose(atom0, torch.zeros(3))
    assert atom1[0].item() == approx(atom1[1].item())
    assert atom1[2].item() == approx(0)

    mode1 = freq_modes1.modes[1]
    atom0, atom1 = mode1
    assert torch.allclose(atom0, torch.zeros(3))
    assert atom1[0].item() == approx(-atom1[1].item())
    assert atom1[2].item() == approx(0)

    mode2 = freq_modes1.modes[2]
    atom0, atom1 = mode2
    assert torch.allclose(atom1, torch.zeros(3))
    assert atom0[0].item() == approx(atom0[1].item())
    assert atom0[2].item() == approx(0)

    mode3 = freq_modes1.modes[3]
    atom0, atom1 = mode3
    assert torch.allclose(atom0, torch.zeros(3))
    assert torch.allclose(atom1[:1], torch.zeros(2))
    assert atom1[2].abs().item() > 1e-3

    mode4 = freq_modes1.modes[4]
    atom0, atom1 = mode4
    assert torch.allclose(atom1, torch.zeros(3))
    assert atom0[0].item() == approx(-atom0[1].item())
    assert atom0[2].item() == approx(0)

    mode5 = freq_modes1.modes[5]
    atom0, atom1 = mode5
    assert torch.allclose(atom1, torch.zeros(3))
    assert torch.allclose(atom0[:1], torch.zeros(2))
    assert atom0[2].abs().item() > 1e-3


###############################################################################
# We also support doing vibrational analysis in batch. In case of batch, the
# hessian should have shape `(molecules, 3 * atoms, 3 * atoms)`, and the mass
# should have shape `(molecules, atoms)`. The resulting tensors is just a stack
# of tensors for the single molecule case.
mass_batch = torch.stack([mass, mass])
freq_modes2 = vib.vibrational_analysis(mass_batch, hessian_batch)
print(freq_modes2.angular_frequencies.shape)
print(freq_modes2.angular_frequencies)
print(freq_modes2.modes.shape)
print(freq_modes2.modes)


###############################################################################
# Let's test if the result is a stack of pervious results
def test_vibrational_analysis_batch():
    af1 = freq_modes1.angular_frequencies
    m1 = freq_modes1.modes
    expected_af = torch.stack([af1, af1])
    expected_modes = torch.stack([m1, m1])
    assert torch.allclose(expected_af, freq_modes2.angular_frequencies)
    assert torch.allclose(expected_modes, freq_modes2.modes)


###############################################################################
# In the case when the batch are isomers (i.e. have the same masses), it is also
# possible to just specify the mass as shape `(atoms,)` and it will broadcast
# automatically.
freq_modes3 = vib.vibrational_analysis(mass, hessian_batch)
print(freq_modes3.angular_frequencies.shape)
print(freq_modes3.angular_frequencies)
print(freq_modes3.modes.shape)
print(freq_modes3.modes)


###############################################################################
# Let's test if we still have the same result
def test_vibrational_analysis_batch_broadcast():
    assert torch.allclose(freq_modes2.angular_frequencies, freq_modes3.angular_frequencies)
    assert torch.allclose(freq_modes2.modes, freq_modes3.modes)


###############################################################################
# Now let's run all the tests
if __name__ == '__main__':
    pytest.main([sys.argv[0], '-v'])
