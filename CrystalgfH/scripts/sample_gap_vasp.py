import argparse
import torch
from pymatgen.io.cif import CifWriter
from sample_api import construct_dataset_from_json, construct_dataset_from_syminfo, generate_structures_from_dataset
import os
from pymatgen.analysis.structure_matcher import StructureMatcher
import argparse
import torch
from pymatgen.io.cif import CifWriter
from sample_api import construct_dataset_from_vasp_format, construct_dataset_from_vasp_gap, construct_dataset_from_syminfo, generate_structures_from_dataset, diffusion, get_pymatgen
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
import json

def main(args):

    with open(args.json_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    tar_dir = args.save_path
    os.makedirs(tar_dir, exist_ok=True)

    if args.json_file != '':
        dataset = construct_dataset_from_vasp_gap(args.json_file)

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
        
    formula_list = []
    gap_list = []
    spg_list = []
    for data in dataset:
        formula_list.append(str(data['formula']))
        gap_list.append(str(data['band_gap']))
        spg_list.append(int(data['spacegroup']))
    
    structure_list = generate_structures_from_dataset(args.model_path, dataset, args.batch_size, args.step_lr)
    #print(structure_list)

    print("Saving structures.")
    print('length_structure_list:', len(structure_list))
    for i,structure in enumerate(structure_list):
        tar_file = os.path.join(tar_dir, f"%05d_{formula_list[i]}_{spg_list[i]}_{gap_list[i]}.cif"%i)
        if structure is not None:
            writer = CifWriter(structure)
            all_data[i]['cif'] = str(writer)
            writer.write_file(tar_file)
        else:
            print(f"{i+1} Error Structure.")
    
    with open("./llama_element_gap_2diff.json", 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent = 7, ensure_ascii=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--spacegroup', default=0, type=int)
    parser.add_argument('--wyckoff_letters', default='', type=str)
    parser.add_argument('--atom_types', default='', type=str)
    parser.add_argument('--json_file', default='', type=str)

    args = parser.parse_args()

    main(args)
