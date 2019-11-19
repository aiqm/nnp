"""
SO(3) Lie Group Operations
==========================

The module ``nnp.so3`` contains tools to rotate point clouds in 3D space.
"""
###############################################################################
# Let's first import all the packages we will use:
import torch
from scipy.linalg import expm as scipy_expm
from torch import Tensor


###############################################################################
# The following function implements rotation along an axis passing the origin.
# Rotation in 3D is not as trivial as in 2D. Here we start from the equation of
# motion. Let :math:`\vec{n}` be the unit vector pointing to
# direction of the axis, then for an infinitesimal rotation, we have
#
# .. math::
#   \frac{\mathrm{d}\vec{r}}{\mathrm{d}\theta}
#       = \vec{n} \times \vec{r}
#
# That is
#
# .. math::
#   \frac{\mathrm{d}r_i}{\mathrm{d}\theta} = \epsilon_{ijk} n_j r_k
#
# where :math:`\epsilon_{ijk}` is the Levi-Civita symbol, let
# :math:`W_{ik}=\epsilon_{ijk} n_j`, then the above equation becomes
# a matrix equation:
#
# .. math::
#   \frac{\mathrm{d}\vec{r}}{\mathrm{d}\theta} = W \cdot \vec{r}
#
# It is not hard to see that :math:`W` is a skew-symmetric matrix.
# From the above equation and the knowledge of linear algebra,
# matrix Lie algebra/group, it is not hard to see that the set of
# all rotation operations along the axis :math:`\vec{n}` is a one
# parameter Lie group. And the skew-symmetric matrices together
# with standard matrix commutator is a Lie algebra.This Lie group
# and Lie algebra is connected by the exponential map. See Wikipedia
# `Exponential map (Lie theory)`_ for more detail.
#
# .. _Exponential map (Lie theory):
#   https://en.wikipedia.org/wiki/Exponential_map_(Lie_theory)
#
# So it is easy to tell that:
#
# .. math::
#   \vec{r}\left(\theta\right) = \exp \left(\theta W\right) \cdot
#       \vec{r}\left(0\right)
#
# where :math:`\vec{r}\left(0\right)` is the initial coordinates,
# and :math:`\vec{r}\left(\theta\right)` is the final coordinates
# after rotating :math:`\theta`.
#
# To implement, let's first define the Levi-Civita symbol:
levi_civita = torch.zeros(3, 3, 3)
levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1
levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1


###############################################################################
# PyTorch does not have matrix exp, let's implement it here using scipy
def expm(matrix: Tensor) -> Tensor:
    # TODO: remove this part when pytorch support matrix_exp
    ndarray = matrix.detach().cpu().numpy()
    return torch.from_numpy(scipy_expm(ndarray)).to(matrix)


###############################################################################
# Now we are ready to implement the :math:`\exp \left(\theta W\right)`
def rotate_along(axis: Tensor) -> Tensor:
    r"""Compute group elements of rotating along an axis passing origin.

    Arguments:
        axis: a vector (x, y, z) whose direction specifies the axis of the rotation,
            length specifies the radius to rotate, and sign specifies clockwise
            or anti-clockwise.

    Return:
        the rotational matrix :math:`\exp{\left(\theta W\right)}`.
    """
    W = torch.einsum('ijk,j->ik', levi_civita.to(axis), axis)
    return expm(W)
