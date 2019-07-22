## Coin dataset

My euro coin dataset for image classification experiments. Front only.

### Structure

```
coin-dataset
├── original
│   ├── 10c [567 entries]
│   ├── 1c [293 entries]
│   ├── 1e [547 entries]
│   ├── 20c [335 entries]
│   ├── 2c [376 entries]
│   ├── 2e [633 entries]
│   ├── 50c [446 entries]
│   └── 5c [481 entries]
├── raw
│   ├── raw1
│   │   ├── 10c_8
│   │   │   ├── IMG_20190707_010612.jpg
│   │   │   ├── IMG_20190707_010616.jpg
│   ...
├── extract_coins.py
├── modify_dataset.py
├── README.md
└── split_dataset.py
```

With `extract_coins.py`, pictures found in `raw` folder (not uploaded) are analyzed, and coins extracted. No elaboration is done on the images.

With `modify_dataset.py` the dataset is elaborated and standardized, there are several options:
* equalize the images using CLAHE on L channel (Lab)
* mask the background
* resize the images

Run `modify_dataset.py -h` for a full list of options.

The dataset can be split in train/validation/test sets using `split_dataset.py`.
