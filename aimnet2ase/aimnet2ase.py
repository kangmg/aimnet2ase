from aimnet2ase.calculators.calculator import AIMNet2Calculator
import torch
from importlib.util import find_spec
import os
import ase
from ase.optimize import BFGS
from IPython.display import clear_output
from ase import Atoms
from ase.io import read
from io import StringIO
import numpy as np


models = lambda : print("""
- AIMNet2 models

    .____________.__________________________.
    |  Abbr.     |      Full Name           | 
    +____________+__________________________+ 
    |  b973c     | aimnet2_b973c_ens.jpt    | 
    |  wb97m-d3  | aimnet2_wb97m-d3_ens.jpt | 
    +____________+__________________________+ 

[Note] 
- Both `Abbr.' and `Full Name` are supported.
""")


abbr = {"b973c"     :  "aimnet2_b973c_ens.jpt",
        "wb97m-d3"  :  "aimnet2_wb97m-d3_ens.jpt"}


def load_model(model:str)->torch.jit.ScriptModule:
  """
  Description
  -----------
  loads model in "./models/"

  Parameters
  ----------
    - model(str) : Name of aimnet2 model. Default is b973c. \n
  Check available models via `aimnet2ase.models()`

  Returns
  -------
    - torch model (torch.jit.ScriptModule)
  """
  # model Directory
  packageDir = find_spec("aimnet2ase").submodule_search_locations[0]
  modelsDir = os.path.join(packageDir, "models")

  # load model 
  if abbr.get(model):
    modelPath = os.path.join(modelsDir, abbr[model])
    return torch.jit.load(modelPath)
  elif model in abbr.values():
    modelPath = os.path.join(modelsDir, model)
    return torch.jit.load(modelPath)
  else:
    assert os.path.exists(model), "model not found"
    return torch.jit.load(model)



def aimnet2_get_energy(xyz_string:str, charge:int, model:str)->float:
  """
  Description
  -----------
  get aimnet2 potential energy in eV

  Parameters
  ----------
  - xyz_string (str) : xyz format string
  - charge (int) : molecular total chage
  - model (str) : AIMNet2 ML potential model

  Supported models
  ----------------
  - b973c      : RKS B97-3c     | H B C N O F Si P S Cl As Se Br I
  - wb97m-d3   : RKS wB97M-D3   | H B C N O F Si P S Cl As Se Br I

  Returns
  -------
  - energy in eV (float) 
  """
  # convert xyz format --> ase.Atoms
  mol = ase.io.read(StringIO(xyz_string), format="xyz")
  
  # check compatibility between the AIMNet2 model and the elements in the molecule
  supporting_elements = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"}

  elements_in_mol = set(mol.get_chemical_symbols())
  unsupported_elements = elements_in_mol - supporting_elements
  if len(unsupported_elements) != 0:
    print(f"{unsupported_elements} are not compatible with the AIMNet2")
    return None

  # AIMNet2 charge compatibility
  if charge not in [-2, -1, 0, 1, 2]:
    print(f"AIMNet2 support not {charge} charged species")
    return None

  # set AIMNet2 calculator
  Calculator = AIMNet2Calculator(load_model(model),charge=charge)
  mol.calc = Calculator

  potential_energy = mol.get_potential_energy()
 
  return potential_energy



def aimnet2_get_force(xyz_string:str, charge:int, model:str)->np.array:
  """
  Description
  -----------
  get aimnet2 force in eV/A

  Parameters
  ----------
  - xyz_string (str) : xyz format string
  - charge (int) : molecular total chage
  - model (str) : AIMNet2 ML potential model

  Supported models
  ----------------
  - b973c      : RKS B97-3c     | H B C N O F Si P S Cl As Se Br I
  - wb97m-d3   : RKS wB97M-D3   | H B C N O F Si P S Cl As Se Br I

  Returns
  -------
  - force array in eV/A 
  """
  # convert xyz format --> ase.Atoms
  mol = ase.io.read(StringIO(xyz_string), format="xyz")
  
  # check compatibility between the AIMNet2 model and the elements in the molecule
  supporting_elements = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"}

  elements_in_mol = set(mol.get_chemical_symbols())
  unsupported_elements = elements_in_mol - supporting_elements
  if len(unsupported_elements) != 0:
    print(f"{unsupported_elements} are not compatible with the AIMNet2")
    return None

  # AIMNet2 charge compatibility
  if charge not in [-2, -1, 0, 1, 2]:
    print(f"AIMNet2 support not {charge} charged species")
    return None

  # set AIMNet2 calculator
  Calculator = AIMNet2Calculator(load_model(model),charge=charge)
  mol.calc = Calculator

  potential_force = mol.get_forces()
 
  return potential_force


def aimnet2_optimize(xyz_string:str, charge:int, model:str, clear_log=True)->str:
  """
  Description
  -----------
  aimnet2 geometry optimize 함수

  Parameters
  ----------
  - xyz_string (str) : xyz format string
  - charge (int) : molecular total chage
  - model (str) : AIMNet2 ML potential model
  - clear_log (bool) : clear optimization logging

  Supported models
  ----------------
  - b973c      : RKS B97-3c     | H B C N O F Si P S Cl As Se Br I
  - wb97m-d3   : RKS wB97M-D3   | H B C N O F Si P S Cl As Se Br I

  Returns
  -------
  - optimized xyz (str)
  """
  # convert xyz format --> ase.Atoms
  mol = ase.io.read(StringIO(xyz_string), format="xyz")
  
  # check compatibility between the AIMNet2 model and the elements in the molecule
  supporting_elements = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"}

  elements_in_mol = set(mol.get_chemical_symbols())
  unsupported_elements = elements_in_mol - supporting_elements
  if len(unsupported_elements) != 0:
    print(f"{unsupported_elements} are not compatible with the AIMNet2")
    return None

  # AIMNet2 charge compatibility
  if charge not in [-2, -1, 0, 1, 2]:
    print(f"AIMNet2 support not {charge} charged species")
    return None

  # set AIMNet2 calculator
  Calculator = AIMNet2Calculator(load_model(model),charge=charge)
  mol.calc = Calculator

  # geometry optimize
  optimize = BFGS(mol)
  optimize.run()

  # get xyz format string
  with StringIO() as output:
    ase.io.write(output, mol, format="xyz")
    opt_xyz = output.getvalue()

  # clear cell output
  if clear_log:
    clear_output()
  return opt_xyz
