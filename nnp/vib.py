"""
Vibrational Analysis
====================

The module ``nnp.vib`` contains tools to compute analytical hessian
and do vibrational analysis.
"""
###############################################################################
# Let's first import all the packages we will use:
import torch
from torch import Tensor
from typing import NamedTuple, Optional


###############################################################################
# With ``torch.autograd`` automatic differentiation engine, computing analytical
# hessian is very simple: Just compute the gradient of energies with respect
# to forces, and then compute the gradient of each element of forces with respect
# to coordinates again.
#
# The ``torch.autograd.grad`` returns a list of ``Optional[Tensor]``. To be
# compatible with ``torch.jit``, we need to use assert statement to allow
# ``torch.jit`` to do `type refinement`_.
#
# .. _type refinement:
#   https://pytorch.org/docs/stable/jit.html#optional-type-refinement
def _get_derivatives_not_none(x: Tensor, y: Tensor, retain_graph: Optional[bool] = None,
                              create_graph: bool = False) -> Tensor:
    ret = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph)[0]
    assert ret is not None
    return ret


def hessian(coordinates: Tensor, energies: Optional[Tensor] = None,
            forces: Optional[Tensor] = None) -> Tensor:
    """Compute analytical hessian from the energy graph or force graph.

    Arguments:
        coordinates: Tensor of shape `(molecules, atoms, 3)` or `(atoms, 3)`
        energies: Tensor of shape `(molecules,)`, or scalar, if specified,
            then `forces` must be `None`. This energies must be computed
            from `coordinates` in a graph.
        forces: Tensor of shape `(molecules, atoms, 3)` or `(atoms, 3)`,
            if specified, then `energies` must be `None`. This forces must
            be computed from `coordinates` in a graph.

    Returns:
        Tensor of shape `(molecules, 3 * atoms, 3 * atoms)` or `(3 * atoms, 3 * atoms)`
    """
    if energies is None and forces is None:
        raise ValueError('Energies or forces must be specified')
    if energies is not None and forces is not None:
        raise ValueError('Energies or forces can not be specified at the same time')
    if forces is None:
        assert energies is not None
        forces = -_get_derivatives_not_none(coordinates, energies, create_graph=True)
    flattened_force = forces.flatten(start_dim=-2)
    force_components = flattened_force.unbind(dim=-1)
    return -torch.stack([
        _get_derivatives_not_none(coordinates, f, retain_graph=True).flatten(start_dim=-2)
        for f in force_components
    ], dim=-1)


###############################################################################
# Below are helper functions to compute vibrational frequencies and normal modes.
# The normal modes and vibrational frquencies satisfies the following equation.
#
# .. math::
#   H q = \omega^2 T q
#
# where :math:`H` is the Hessian matrix, :math:`q` is the normal coordinates, and
#
# .. math::
#   T=\left[
#    \begin{array}{ccccccc}
#    m_{1}\\
#    & m_{1}\\
#    &  & m_{1}\\
#    &  &  & m_{2}\\
#    &  &  &  & m_{2}\\
#    &  &  &  &  & m_{2}\\
#    &  &  &  &  &  & \ddots
#    \end{array}
#    \right]
#
# is the mass for each coordinate.
#
# This is a generalized eigen problem, which is not immediately supported by PyTorch
# So we solve this problem through Lowdin diagnolization:
#
# .. math::
#   H q = \omega^2 T q \Longrightarrow H q = \omega^2 T^{\frac{1}{2}} T^{\frac{1}{2}} q
#
# Let :math:`q' = T^{\frac{1}{2}} q`, we then have
#
# .. math::
#   T^{\frac{1}{2}} H T^{\frac{1}{2}} q' = \omega^2 q'
#
# this is a regular eigen problem
class FreqsModes(NamedTuple):
    angular_frequencies: Tensor
    modes: Tensor


def vibrational_analysis(masses: Tensor, hessian: Tensor) -> FreqsModes:
    """Computing the vibrational wavenumbers from hessian.

    Arguments:
        masses: Tensor of shape `(molecules, atoms)` or `(atoms,)`.
        hessian: Tensor of shape `(molecules, 3 * atoms, 3 * atoms)` or
            `(3 * atoms, 3 * atoms)`.

    Returns:
        A namedtuple `(angular_frequencies, modes)` where

        angular_frequencies:
            Tensor of shape `(molecules, 3 * atoms)` or `(3 * atoms,)`
        modes:
            Tensor of shape `(molecules, modes, atoms, 3)` or `(modes, atoms, 3)`
            where `modes = 3 * atoms` is the number of normal modes.
    """
    inv_sqrt_mass = masses.rsqrt().repeat_interleave(3, dim=-1)
    mass_scaled_hessian = hessian * inv_sqrt_mass.unsqueeze(-2) * inv_sqrt_mass.unsqueeze(-1)
    eigenvalues, eigenvectors = torch.symeig(mass_scaled_hessian, eigenvectors=True)
    angular_frequencies = eigenvalues.sqrt()
    modes = (eigenvectors.transpose(-1, -2) * inv_sqrt_mass.unsqueeze(-2))
    new_shape = modes.shape[:-1] + (-1, 3)
    modes = modes.reshape(new_shape)
    return FreqsModes(angular_frequencies, modes)
