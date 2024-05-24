## AIMNet2 with ASE interface.

This repository contains only the minimum files required to use AIMNet2 with ASE. For more details, please refer to the [link](https://github.com/isayevlab/AIMNet2/blob/main/README.md).
  
## Usage

```python
# install this repo
!pip install git+https://github.com/kangmg/aimnet2ase.git

from aimnet2ase import aimnet2_optimize, aimnet2_get_energy

xyz_input = """4

N     -0.045012    1.759938    0.842995
H     -0.081846    1.135119    1.675327
H     -0.095049    2.764070    1.441421
H      0.955910    1.715485    0.238776"""

# optimize xyz string
opt_xyz = aimnet2_optimize(xyz_input, charge=0, model="wb97m-d3")

# calculate optimized structure energy in eV 
opt_energy_in_eV = aimnet2_get_energy(opt_xyz, charge=0, model="wb97m-d3")


print(f"Opt. Energy in eV : {opt_energy_in_eV}\n"

print("Opt. xyz \n", opt_xyz)
```
