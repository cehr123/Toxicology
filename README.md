# Toxicology
This repository hosts our working copy of **ET-Tox** which was adapted from the official
[torchmd/ET-Tox](https://github.com/torchmd/ET-Tox) release. We run our model on UMich Great Lakes cluster. 

---
## Repository Layout

| Path | Description |
- `ET-Tox/train_yaml/`: Ready-to-run experiment configs used for our eval(Tox21, BBBP). Toggle features for edge attention and n-grams via these files.
- `ET-Tox/data/`: Local copy of MoleculeNet/TDC splits.
- `ET-Tox/train.sh`: Example SLURM batch script for the Tox21 weights-only experiment.
- `ET-Tox/output_dir`: Example training logs, metrics, checkpoints (with/without attention, different seeds).
  
---
## Environment Setup

```bash
cd /path/to/Toxicology/ET-Tox
conda env create -f environment.yml   
conda activate et_tox
pip install -e .                      
```

---
## Data
1. Download MoleculeNet/TDC files from
   [Zenodo 7942946](https://zenodo.org/record/7942946).
2. Place into `ET-Tox/data/`:
   - Files containing `mnet` → `ET-Tox/data/MoleculeNet/`
   - Files containing `tdc`  → `ET-Tox/data/TDCTox/`
3. Provided splits (`*_seed*_confs*_*.npz`) already live under
   `ET-Tox/data/MoleculeNet/` for Tox21 + BBBP seeds 1 and 1000.

---

## Running Experiments

### 1. Direct (interactive) run (with GPU) 

The command to run the training code for MoleculeNet datsets is:

```bash
python train.py --conf ./train_yaml/{DATASET}.yaml --log-dir {OUTPUT_DIR}/{DATASET}/seed{SEED#} --seed {SEED#} --splits ./data/MoleculeNet/splits/{DATASET}_seed{SEED#}_confs1_{SPLIT}.npz
```

For TDCTox and ToxBenchmark: 

```bash
python train.py --conf ./train_yaml/{DATASET}.yaml --log-dir {OUTPUT_DIR}/{DATASET}/seed{SEED#} --seed {SEED#} --splits ./data/TDCTox/splits/{DATASET}_seed{SEED#}_confs1_{SPLIT}.npz --dataset-split {SPLIT}
```

### 2. Batch via SLURM

Use `run_tox21_seed1000.sh` as a template. Key flags:

- `#SBATCH --partition=gpu --gres=gpu:1`
- Activate the environment (`conda activate et_tox`)
- sbatch fileNameHere

### 3. Testing pretrained checkpoints

```bash
python train.py \
  --conf ./train_yaml/tox21.yaml \
  --test-run true \
  --test-checkpoint ./output_dir/tox21.ckpt
```

Logs are saved under the chosen `--log-dir`. Recent successful runs are available in the `{OUTPUT_DIR}` folders.

---
## Our Outputs

In our code, we use the Tox21, ToxCast, SIDER, ClinTox, and BBBP datasets from MoleculeNet. The dataset files with the splits and conformer numbers are found at the source in the Data section above. Those files will be under the `MoleculeNet/splits` directory. We also use the Mutagenicity dataset from ToxBenchmark. From the data source above, the files are in the `TDCTox/splits` directory. 

Each dataset version should be labeled as `{DATASET}_seed{SEED#}_confs1_{SPLIT}`. 

To change between the different versions of our model with different components included, edit the `.yaml` files for the corresponding dataset. 

For:
* Model we evaluated in the paper (edge embeddings + attention + no ngrams):
  ```
  model: equivariant-transformer-edge
  use_n_gram: false
  use_edge_attention: true
  ```
* Complete model (edge embeddings + attention + ngrams):
  ```
  model: equivariant-transformer-edge
  use_n_gram: true
  use_edge_attention: true
  ```
* Ablation model with no attention:
  ```
  model: equivariant-transformer-edge
  use_n_gram: false
  use_edge_attention: false
  ```
* Basic EGNN:
  ```
  model: equivariant-transformer
  use_n_gram: false
  use_edge_attention: false
  ```
  
### Evaluation

The only evaluation code we used is in test_models.ipynb, change the parameters in the second code block,
```
    dataset = "MoleculeNet"
    dataset_arg = {"num_conformers": 1, "conformer": "best", "data_version": "geom", "dataset": name}
    dataset_root = "./data/MoleculeNet"
    dataset_split = "random"
    splits_path = f"./data/MoleculeNet/splits/{name}_seed{n}_confs1_random.npz"
```
based on what dataset you are evaluating. 

The second block calculates ROC-AUC for each of the seed you pass to it using these variables at the top of the block:
```
name = "bbbp"
dataset_names = ["1", "42", "100", "500", "1000"] 
output_dir =  "z_output_dir_model" 
rocauc_each = []
```

Running the third block of the notebook will get the mean and standard deviation of what is in `rocauc_each`.

---

## Citation
> Cremer, J.; Sandonas, L. M.; Tkatchenko, A.; Clevert, D.-A.; de Fabritiis, G.  
> *Equivariant Graph Neural Networks for Toxicity Prediction.* ChemRxiv (2023).
