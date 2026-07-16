# Checkpoints

This repository does not include pretrained checkpoints.

Train the models with:

```bash
python scripts/train_coarse_phase.py --config configs/coarse_phase.yaml --data_root /path/to/radiomap --save_dir outputs/coarse_phase
python scripts/train_refine_phase.py --config configs/refine_phase.yaml --data_root /path/to/radiomap --coarse_ckpt outputs/coarse_phase/coarse_best.pth --save_dir outputs/refine_phase
```

Do not commit `.pth`, `.pt`, or other large checkpoint files to Git.
