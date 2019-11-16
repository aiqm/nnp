"""
Rotation in 3D Space
====================

This tutorial demonstrates how rotate a group of points in 3D space.
"""
###############################################################################
# To begin with, we must understand that rotation is a linear transformation.
# The set of all rotations forms the group SO(3). It is a Lie group that each
# group element is described by an orthogonal 3x3 matrix.
#
# That is, if you have two points :math:`\vec{r}_1` and :math:`\vec{r}_2`, and
# you want to rotate the two points along the same axis for the same number of
# degrees, then there is a single orthogonal matrix :math:`R` that no matter
# the value of :math:`\vec{r}_1` and :math:`\vec{r}_2`, their rotation is always
# :math:`R\cdot\vec{r}_1` and :math:`R\cdot\vec{r}_2`.
#
# Let's import libraries first
import math
import torch
import pytest
import sys
import nnp.so3 as so3

###############################################################################
# Let's first take a look at a special case: rotating the three unit vectors
# ``(1, 0, 0)``, ``(0, 1, 0)`` and ``(0, 0, 1)`` along the diagonal for 120
# degree, this should permute these points:
rx = torch.tensor([1, 0, 0])
ry = torch.tensor([0, 1, 0])
rz = torch.tensor([0, 0, 1])
points = torch.stack([rx, ry, rz])

###############################################################################
# Now let's compute the rotation matrix
axis = torch.ones(3) / math.sqrt(3) * (math.pi * 2 / 3)
R = so3.rotate_along(axis)

###############################################################################
# Note that we need to do matrix-vector product for all these unit vectors
# so we need to do a transpose in order to use ``@`` operator.
rotated = (R @ points.float().t()).t()
print(rotated)


###############################################################################
# After this rotation, the three vector permutes: rx->ry, ry->rz, rz->rx. Let's
# programmically check that. This check will be run by pytest later.
def test_rotated_unit_vectors():
    expected = torch.stack([ry, rz, rx]).float()
    assert torch.allclose(rotated, expected, atol=1e-5)


###############################################################################
# Now let's run all the tests
if __name__ == '__main__':
    pytest.main([sys.argv[0]])
