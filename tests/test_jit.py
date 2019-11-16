import torch
import pytest
import sys
# import nnp.so3 as so3
import nnp.pbc as pbc
# import nnp.vib as vib


def test_script():
    # torch.jit.script(so3.rotate_along)
    torch.jit.script(pbc.num_repeats)
    torch.jit.script(pbc.map2central)
    # torch.jit.script(vib.hessian)
    # torch.jit.script(vib.vibrational_analysis)


if __name__ == '__main__':
    pytest.main([sys.argv[0]])
