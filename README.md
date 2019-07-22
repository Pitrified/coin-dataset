## Coin dataset

My euro coin dataset for image classification experiments. Front only.

### Structure

```
coin-dataset
├── original
│   ├── 10c [1025 images]
│   ├── 1c [1150 images]
│   ├── 1e [1027 images]
│   ├── 20c [1058 images]
│   ├── 2c [1005 images]
│   ├── 2e [1041 images]
│   ├── 50c [1031 images]
│   └── 5c [1088 images]
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
