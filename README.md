# Toxicology
This repository hosts our working copy of **ET-Tox** which was adapted from the official
[torchmd/ET-Tox](https://github.com/torchmd/ET-Tox) release. We run our model on UMich Great Lakes cluster. 

---
## Repository Layout

| Path | Description |
- `ET-Tox/train_yaml/`: Ready-to-run experiment configs used for our eval(Tox21, BBBP). Toggle features for edge attention and n-grams via these files.
- `ET-Tox/data/`: Local copy of MoleculeNet/TDC splits.
- `ET-Tox/run_tox21_seed1000.sh`: Example SLURM batch script for the Tox21 weights-only experiment.
- `ET-Tox/z_output_*`: Example training logs, metrics, checkpoints (with/without attention, different seeds).
  
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

Allocate time on supercomputer.

```bash
python train.py \
  --conf ./train_yaml/tox21.yaml \
  --log-dir ./z_output_without_attention/tox21/seed1 \
  --seed 1 \
  --splits ./data/MoleculeNet/tox21_seed1_confs1_random.npz
```

We keep separate log roots (`z_output_without_attention`, `z_output_attention_only`,
`z_output_weights_only`) to compare variants.

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

Logs are saved under the chosen `--log-dir`. Recent successful runs are available in the `z_output_*` folders.

---
## Our Outputs

- `ET-Tox/z_output_without_attention/` – Tox21/BBBP runs with `use_edge_attention: false`.
- `ET-Tox/z_output_attention_only/` – attention-only ablations.
- `ET-Tox/z_output_weights_only/` – weight-only experiments.

To reproduce a run, copy the relevant YAML + split file and reuse the logged
seed, then compare metrics with `metrics.csv`.

---

## Citation
> Cremer, J.; Sandonas, L. M.; Tkatchenko, A.; Clevert, D.-A.; de Fabritiis, G.  
> *Equivariant Graph Neural Networks for Toxicity Prediction.* ChemRxiv (2023).
