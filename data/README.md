# Data

Place the radio-map dataset outside this repository and pass its path with `--data_root`.

Expected structure:

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

Large dataset files should not be committed to Git.
