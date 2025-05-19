import argparse
import torch
from pymatgen.io.cif import CifWriter
from sample_api import construct_dataset_from_json, construct_dataset_from_syminfo, generate_structures_from_dataset
import os
from pymatgen.analysis.structure_matcher import StructureMatcher
import argparse
import torch
from pymatgen.io.cif import CifWriter
from sample_api import construct_dataset_from_json_gap, construct_dataset_from_json_format, construct_dataset_from_syminfo, generate_structures_from_dataset, diffusion, get_pymatgen
import os
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
import os
from pathlib import Path
from eval_utils import load_model, get_crystals_list, lattices_to_params_shape
from torch_geometric.data import Data, Batch, DataLoader
from p_tqdm import p_map
import time
from pymatgen.analysis.structure_matcher import StructureMatcher

def get_X(cif):
  #print(cif)
  cif_list = cif.split('\n')
  V = cif_list[12].split('  ')[1]
  #print(V)
  n = len(cif_list) - cif_list.index(' _atom_site_occupancy') - 2
  #print(n)
  X = (float(V) / int(n))**(1/3)
  return X
  
def main(args):

    
    tar_dir = args.save_path
    os.makedirs(tar_dir, exist_ok=True)

    if args.json_file != '':
        dataset = construct_dataset_from_json_gap(args.json_file)

    else:
        assert args.spacegroup > 0 and args.wyckoff_letters != ''
        if ',' in args.wyckoff_letters:
            wyckoff_letters = args.wyckoff_letters.split(',')
        else:
            wyckoff_letters = args.wyckoff_letters

        if args.atom_types != '':
            atom_types = args.atom_types.split(',')
        else:
            atom_types = None

        dataset = construct_dataset_from_syminfo(args.spacegroup, wyckoff_letters, atom_types)

    volume_list = []
    cif_list = []
    for data in dataset:
      #print(data)
      volume_list.append(float(data['volume']))
      cif_list.append(str(data['cif']))
    #print(volume_list)
    #print(cif_list)
    structure_list = generate_structures_from_dataset(args.model_path, dataset, args.batch_size, args.step_lr)
    #print(structure_list)

    M = 0
    N = 0
    X2_all = 0
    RMSE = 0
    MR = 0
    structureMatcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    
    print("Saving structures.")
    for i,structure in enumerate(structure_list):
        if structure is not None:

            file_name = str(structure.formula.replace(" ",""))
            tar_file = os.path.join(tar_dir, f"%s.cif"%file_name)
            writer = CifWriter(structure)

            cif_origin = cif_list[i]
            structure_origin = Structure.from_str(cif_origin, fmt='cif')
            
            Match = structureMatcher.fit(structure, structure_origin)
            N += 1
            if Match == True:
              M += 1
              X_origin = get_X(cif_list[i])
              X_new = get_X(str(writer))
              X = (X_origin - X_new) ** 2
              X2_all += X
              RMSE = (X2_all / M) ** (1/2)
              MR = M/N
              print('MR:', MR)
              print('RMSE:', RMSE)
        else:
            print(f"{i+1} Error Structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--spacegroup', default=0, type=int)
    parser.add_argument('--wyckoff_letters', default='', type=str)
    parser.add_argument('--atom_types', default='', type=str)
    parser.add_argument('--json_file', default='', type=str)

    args = parser.parse_args()

    main(args)
