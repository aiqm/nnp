"""
Periodic Boundary Conditions
============================

This file contains
"""

import torch
from torch import Tensor


def map2central(cell: Tensor, coordinates: Tensor, pbc: Tensor) -> Tensor:
    """Map atoms outside the unit cell into the cell using PBC.

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:

            .. code-block:: python

                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])

        coordinates (:class:`torch.Tensor`): Tensor of shape ``(atoms, 3)``
            or ``(molecules, atoms, 3)``.
        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.
    Returns:
        :class:`torch.Tensor`: coordinates of atoms mapped back to unit cell.
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
