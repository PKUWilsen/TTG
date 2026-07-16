# TTG

This repository contains the open-source implementation of **TTG**, a two-phase radio map estimation framework.

The code is organized around two phases:

1. **Adapt Phase**: an Attention U-Net predicts a coarse radio map from the obstacle layout and sparse RSS samples, and the coarse map is further used to construct enhanced samples and transmitter-position cues.
2. **Refine Phase**: a diffusion-based refinement model improves the coarse map using enhanced sparse conditioning and DDIM sampling.

<p align="center">
  <img src="assets/framework_overview.png" alt="TTG framework overview" width="900">
</p>

Inside the Refine Phase, inference uses a **three-stage DDIM guidance** strategy:

- **Stage I**: high-threshold coverage for structural rudiment generation.
- **Stage II**: free-form diffusion for prototype generation.
- **Stage III**: truth-anchor guidance using real sparse samples.

<p align="center">
  <img src="assets/guidance_process.png" alt="Three-stage conditional guidance process" width="900">
</p>

The terms *phase* and *stage* are used deliberately: the whole TTG pipeline has two phases, while only the refine-phase sampler has three guidance stages.

## Repository structure

```text
TTG/
├── configs/
│   ├── coarse_phase.yaml
│   ├── refine_phase.yaml
│   └── eval.yaml
├── ttg/
│   ├── datasets/
│   ├── models/
│   ├── diffusion/
│   └── utils/
├── scripts/
│   ├── train_coarse_phase.py
│   ├── train_refine_phase.py
│   ├── eval_refine_phase.py
│   └── visualize.py
├── wash.py
├── data/README.md
├── checkpoints/README.md
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n ttg python=3.10 -y
conda activate ttg
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if the default pip installation is not suitable for your machine.

## Dataset

The code expects a RadioUNet-style radio-map dataset with MATLAB files:

```text
radiomap/
├── buildings_position/
├── receivedpower_1750MHz_mat/
├── receivedpower_2750MHz_mat/
├── receivedpower_3750MHz_mat/
├── receivedpower_4750MHz_mat/
├── receivedpower_5750MHz_mat/
└── stations_position.txt
```

Each `receivedpower_*MHz_mat/` directory contains dense RSS maps. `buildings_position/` contains the corresponding building maps. `stations_position.txt` stores base-station coordinates used by the dataset loader.


Default split ranges follow the original runnable TTG code: train `[0, 1576)`, validation `[1300, 1576)`, and test `[1676, 1776)`. Edit these ranges only if you intentionally want to run a different split.

Set the dataset path by command line:

```bash
--data_root /path/to/radiomap
```

or edit `data.data_root` in the YAML config files.

## Dataset cleaning utility

`wash.py` is kept unchanged from the original runnable TTG code. It directly deletes samples whose multi-frequency RSS map shapes are inconsistent or whose map size is no larger than 256 x 256. It also retains the original hard-coded `dataset_path`, so review and edit that path carefully before running it.

```bash
python wash.py
```

## Adapt Phase

Train the coarse estimator:

```bash
python scripts/train_coarse_phase.py \
  --config configs/coarse_phase.yaml \
  --data_root /path/to/radiomap \
  --save_dir outputs/coarse_phase
```

The best checkpoint is saved as:

```text
outputs/coarse_phase/coarse_best.pth
```

## Refine Phase

Train the diffusion refinement model using the trained coarse estimator:

```bash
python scripts/train_refine_phase.py \
  --config configs/refine_phase.yaml \
  --data_root /path/to/radiomap \
  --coarse_ckpt outputs/coarse_phase/coarse_best.pth \
  --save_dir outputs/refine_phase
```

The best refinement checkpoint is saved as:

```text
outputs/refine_phase/refine_best.pth
```

## Evaluation

Evaluate the full two-phase TTG pipeline:

```bash
python scripts/eval_refine_phase.py \
  --config configs/eval.yaml \
  --data_root /path/to/radiomap \
  --coarse_ckpt outputs/coarse_phase/coarse_best.pth \
  --refine_ckpt outputs/refine_phase/refine_best.pth \
  --save_dir outputs/eval
```

The current evaluation pipeline assumes `batch_size=1`, because sparse enhancement and sparse-point election are computed per test sample.

## Checkpoints

This repository does **not** include pretrained checkpoints. Please train the coarse estimator and refinement model following the provided scripts.

## Notes

- The main training/evaluation scripts do not hard-code local machine paths.
- `wash.py` is a legacy utility kept unchanged from the original runnable code and still contains the original hard-coded path.
- The YAML files contain placeholder paths and default hyperparameters.
- The coarse/refine terminology is used for the two TTG phases.
- The word stage is reserved for the three-stage DDIM guidance inside the Refinement Phase.

## Citation

Add the BibTeX entry of the TTG paper here when available.

## License

This project is released under the MIT License. See `LICENSE` for details.
