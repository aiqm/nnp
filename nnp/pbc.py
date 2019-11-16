"""
Periodic Boundary Conditions
============================

The module ``nnp.pbc`` contains tools to deal with periodic boundary conditions.
"""
###############################################################################
# Let's first import all the packages we will use:
import torch
from torch import Tensor


###############################################################################
# The following function computes the number of repeats required to capture
# all the neighbor atoms within the cutoff radius.
def num_repeats(cell: Tensor, pbc: Tensor, cutoff: float) -> Tensor:
    """Compute the number of repeats required along each cell vector to make
    the original cell and repeated cells together form a large enough box
    so that for each atom in the original cell, all its neighbor atoms within
    the given cutoff distance are contained in the big box.

    Arguments:
        cell: tensor of shape ``(3, 3)`` of the three vectors defining unit cell:

            .. code-block:: python

                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])

        pbc: boolean vector of size 3 storing if pbc is enabled for that direction.
        cutoff: the cutoff inside which atoms are considered as pairs

    Returns:
        numbers of repeats required to make the repeated box contains all the neighbors
        of atoms in the ceter cell
    """
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    result = torch.ceil(cutoff * inv_distances).to(torch.long)
    return torch.where(pbc, result, torch.tensor(0))


def map2central(cell: Tensor, coordinates: Tensor, pbc: Tensor) -> Tensor:
    """Map atoms outside the unit cell into the cell using PBC.

    Arguments:
        cell: tensor of shape ``(3, 3)`` of the three vectors defining unit cell:

            .. code-block:: python

                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])

        coordinates: Tensor of shape ``(atoms, 3)`` or ``(molecules, atoms, 3)``.
        pbc: boolean vector of size 3 storing if pbc is enabled for that direction.

    Returns:
        coordinates of atoms mapped back to unit cell.
    """
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    inv_cell = torch.inverse(cell)
    coordinates_cell = coordinates @ inv_cell
    # Step 2: wrap cell coordinates into [0, 1)
    coordinates_cell -= coordinates_cell.floor() * pbc.to(coordinates_cell.dtype)
    # Step 3: convert from cell coordinates back to standard cartesian
    # coordinate
    return coordinates_cell @ cell
