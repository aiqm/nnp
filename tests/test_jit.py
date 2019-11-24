import torch
import pytest
import sys
# import nnp.so3 as so3
import nnp.pbc as pbc
import nnp.vib as vib

TORCH_TOO_OLD = torch.__version__ < "1.4.0.dev20191123"


@pytest.mark.skipif(TORCH_TOO_OLD, reason="JIT compatibility requires new PyTorch")
def test_script():
    # torch.jit.script(so3.rotate_along)
    torch.jit.script(pbc.num_repeats)
    torch.jit.script(pbc.map2central)
    torch.jit.script(vib.hessian)
    torch.jit.script(vib.vibrational_analysis)


if __name__ == '__main__':
    pytest.main([sys.argv[0], '-v'])
