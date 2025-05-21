# CrystalGF
A Generation Framework with Strict Constraints for Crystal Materials Design

### CrystalgfG

CrystalgfG use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine tune the constraint generator.

#### Dependencies and Setup

Install LLaMA-Factory project.

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
```

Building a python environment using conda and run the following command to install the environment:

```
conda create -n llamafactory python=3.10
conda activate llamafactory
pip install -e ".[torch,metrics]"
pip install -e ".[deepspeed,modelscope]"
```

Verify that the installation is successful

```
llamafactory-cli env
```

#### Fine-tuning

Replace the YAML file in `lora_yaml` and `example/inference/`

```
nohup torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 7001 src/train.py lora_yaml/<yaml_name>.yaml  > <log_name>.log
```

Data and pre-trained lora weights are provided [here](https://www.modelscope.cn/models/chachapro/CrystalGF).

#### inference

```
nohup llamafactory-cli train examples/inference/<yaml_name>.yaml > <log_name>.log
```

### CrystalgfH

CrystalgfH is improved on the basis of [DiffCSP++](https://github.com/jiaor17/DiffCSP-PP), and implements the generation of crystal structures with elemental compositon, symmetry information and target material property.


#### Dependencies and Setup

Building a python environment using conda and run the following command to install the environment:

```
conda env create -f environment.yml
```

Specify the following variables in the `.env` file.

```
PROJECT_ROOT: the absolute path of CrystalgfH
HYDRA_JOBS: the absolute path to save hydra outputs
WABDB_DIR: the absolute path to save wabdb outputs
```

#### Training

For the gap-based task
```
nohup python diffcsp/run_gap.py data=<dataset>  expname=<expname> > ./<logname>.log
```

For the formation-based task
```
nohup python diffcsp/run_format.py data=<dataset> expname=<expname> > ./<logname>.log
```
Data and pre-trained checkpoints are provided [here](https://www.modelscope.cn/models/chachapro/CrystalGF).

#### Evaluation

Calculate the matching rate (MR) and root mean square error (RMSE) while generating the structures.

```
nohup python scripts/sample_format.py --model_path <model_path> --save_path <save_path> --json_file <json_file> > ./<log_name>.log
nohup python scripts/sample_gap.py --model_path <model_path> --save_path <save_path> --json_file <json_file> > ./<log_name>.log
```

Json file can be

`example/example.json`:

```
[
      {
            "formula": "Sc2Ni6",
            "n_atom": 8,
            "gap": 0.0,
            "spacegroup_number": 194,
            "atom_types": [
                  "Sc",
                  "Ni"
            ],
            "wyckoff_letters": [
                  "2d",
                  "6h"
            ]
      },
      {
            "formula": "Ba1Nd1Co2O5",
            "n_atom": 9,
            "gap": 0.0,
            "spacegroup_number": 123,
            "atom_types": [
                  "Ba",
                  "Nd",
                  "Co",
                  "O",
                  "O"
            ],
            "wyckoff_letters": [
                  "1d",
                  "1c",
                  "2g",
                  "4i",
                  "1b"
            ]
      }
]
```

Generate structure to prepare for subsequent VASP calculations
```
nohup python scripts/sample_format_vasp.py --model_path <model_path> --save_path <save_path> --json_file <json_file> > ./<log_name>.log
nohup python scripts/sample_gap_vasp.py --model_path <model_path> --save_path <save_path> --json_file <json_file> > ./<log_name>.log
```



