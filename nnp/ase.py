# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import torch
import ase.calculators.calculator


def map2central(cell, coordinates, pbc):
    """Map atoms outside the unit cell into the cell using PBC.
    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:
            .. code-block:: python
                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])
        coordinates (:class:`torch.Tensor`): Tensor of shape
            ``(molecules, atoms, 3)``.
        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.
    Returns:
        :class:`torch.Tensor`: coordinates of atoms mapped back to unit cell.
    """
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    inv_cell = torch.inverse(cell)
    coordinates_cell = torch.matmul(coordinates, inv_cell)
    # Step 2: wrap cell coordinates into [0, 1)
    coordinates_cell -= coordinates_cell.floor() * pbc.to(coordinates_cell.dtype)
    # Step 3: convert from cell coordinates back to standard cartesian
    # coordinate
    return torch.matmul(coordinates_cell, cell)


class Calculator(ase.calculators.calculator.Calculator):
    """ASE Calculator that wraps a neural network potential

    Arguments:
        model (:class:`torch.nn.Module`): neural network potential model
            that convert coordinates into energies.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, model, overwrite=False):
        super(Calculator, self).__init__()
        self.model = model
        self.overwrite = overwrite

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super(Calculator, self).calculate(atoms, properties, system_changes)
        cell = torch.from_numpy(self.atoms.get_cell(complete=True))
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool)
        coordinates = torch.from_numpy(self.atoms.get_positions()).requires_grad_('forces' in properties)
        pbc_enabled = pbc.any().item()

        if pbc_enabled:
            coordinates = map2central(cell, coordinates, pbc)
            if self.overwrite and atoms is not None:
                atoms.set_positions(coordinates.detach().cpu().numpy())

        if 'stress' in properties:
            scaling = torch.eye(3, requires_grad=True, dtype=self.dtype, device=self.device)
            coordinates = coordinates @ scaling
            cell = cell @ scaling
        energy = self.model(atoms.get_chemical_symbols(), coordinates, cell, pbc)

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        if 'forces' in properties:
            forces = -torch.autograd.grad(energy, coordinates)[0]
            self.results['forces'] = forces.cpu().numpy()

        if 'stress' in properties:
            volume = self.atoms.get_volume()
            stress = torch.autograd.grad(energy.squeeze(), scaling)[0] / volume
            self.results['stress'] = stress.cpu().numpy()
