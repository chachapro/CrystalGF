# CrystalGF
A Generation Framework with Strict Constraints for Crystal Materials Design

### CrystalgfG


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

#### Evaluation

Calculate the matching rate (MR) and root mean square error (RMSE) while generating the structures.

```
nohup python scripts/sample_format.py --model_path <model_path> --save_path <save_path> --json_file <json_file> > ./<log_name>.log
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




