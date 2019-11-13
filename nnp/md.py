"""
Molecular Dynamics
==================

The module `nnp.md` provide tools to run molecular dynamics with a potential
defined by PyTorch.
"""

import torch
from torch import Tensor
import ase.calculators.calculator
from nnp import pbc
from typing import Callable, Sequence


class Calculator(ase.calculators.calculator.Calculator):
    """ASE Calculator that wraps a neural network potential

    Arguments:
        func (callable): A fucntion that .
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, func: Callable[[Sequence[str], Tensor, Tensor, Tensor], Tensor],
                 overwrite: bool = False):
        super(Calculator, self).__init__()
        self.func = func
        self.overwrite = overwrite

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super(Calculator, self).calculate(atoms, properties, system_changes)
        coordinates = torch.from_numpy(self.atoms.get_positions()).requires_grad_('forces' in properties)
        cell = coordinates.new_tensor(self.atoms.get_cell(complete=True).array)
        pbc_ = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool)
        pbc_enabled = pbc_.any().item()

        if pbc_enabled:
            coordinates = pbc.map2central(cell, coordinates, pbc_)

        if 'stress' in properties:
            scaling = torch.eye(3, requires_grad=True)
            coordinates = coordinates @ scaling
            cell = cell @ scaling

        energy = self.func(atoms.get_chemical_symbols(), coordinates, cell, pbc_)

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        if 'forces' in properties:
            forces = -torch.autograd.grad(energy, coordinates)[0]
            self.results['forces'] = forces.cpu().numpy()

        if 'stress' in properties:
            volume = self.atoms.get_volume()
            stress = torch.autograd.grad(energy, scaling)[0] / volume
            self.results['stress'] = stress.cpu().numpy()
