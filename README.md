# sigcore — Signature Verification Toolkit

A deep-learning pipeline for offline handwritten signature feature extraction and writer-dependent verification.

---

## Package layout

```
sigcore/
├── inknet/
│   ├── encoder/
│   │   ├── model.py        # InkEncoder, InkEncoderThin, InkEncoderCompact
│   │   └── loader.py       # AugmentedDataset, compute_embeddings
│   ├── preprocessing/
│   │   └── transform.py    # prepare_image, center_on_canvas, …
│   ├── datasets/
│   │   ├── base.py         # SignatureSource abstract base class
│   │   └── util.py         # load_dataset, build_dataset, get_subset, …
│   └── verification/
│       ├── metrics.py      # EER, AUC, FAR/FRR
│       ├── data.py         # train/test splits, per-user set construction
│       └── training.py     # SVM-based writer-dependent classifier
├── train.py                # Feature extractor training script
├── eval_model.py           # Evaluate a .pth checkpoint on a CSV pair-list
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
cd sigcore
pip install -r requirements.txt
# (optional) install as a package so inknet is importable anywhere:
pip install -e .
```

---

## Quick start — evaluate a pre-trained checkpoint

```bash
python eval_model.py \
    --model-path /path/to/model_best.pth \
    --data-dir   /path/to/test_images \
    --csv        /path/to/test_data.csv
```

The CSV must have lines of the form `image1_rel_path,image2_rel_path,label`
where `label=0` means a genuine pair and `label=1` means a forgery pair.

Expected output:

```
Device: cuda
Checkpoint loaded.

Pairs: 4000
  processed 100/4000
  ...
AUC  : 0.9821
EER  : 0.0634
Threshold: 0.7412
```

---

## Training from scratch

### 1. Prepare dataset

Implement a `SignatureSource` subclass for your dataset (see `inknet/datasets/base.py`) and run:

```python
from inknet.datasets.util import build_dataset
from my_dataset import MyDataset

build_dataset(MyDataset('/data/raw'), save_path='/data/processed.npz', img_size=(150, 220))
```

### 2. Train the feature extractor

```bash
python train.py \
    --dataset-path /data/processed.npz \
    --model        inkencoder \
    --logdir       ./runs/exp1 \
    --users        350 881 \
    --epochs       60 \
    --lr           0.001 \
    --batch-size   32
```

To also train with forgery supervision (InkEncoder-F):

```bash
python train.py \
    --dataset-path /data/processed.npz \
    --model  inkencoder \
    --logdir ./runs/exp_forg \
    --forg \
    --lamb   0.95 \
    --loss-type L2
```

Checkpoints are written to `--logdir` as `model_best.pth` and `model_last.pth`.
The `.pth` format is a 3-tuple: `(base_state_dict, cls_head_state_dict, forg_head_state_dict)`.

### 3. Resume training

```bash
python train.py \
    --dataset-path /data/processed.npz \
    --model   inkencoder \
    --logdir  ./runs/exp1_resumed \
    --checkpoint ./runs/exp1/model_best.pth \
    --epochs  30 \
    --lr      0.0001
```

---

## Writer-dependent (WD) evaluation

After extracting features with a trained checkpoint you can train and benchmark
per-user SVM classifiers:

```python
import torch, numpy as np
from inknet.encoder.model import InkEncoder
from inknet.encoder.loader import compute_embeddings
from inknet.datasets.util import load_dataset, get_subset
from inknet.verification.training import run_fold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sd, _, _ = torch.load('runs/exp1/model_best.pth', map_location=device)
model = InkEncoder().to(device).eval()
model.load_state_dict(sd)

x, y, yforg, _, _ = load_dataset('/data/processed.npz')

def process_fn(batch):
    return model(batch[0].to(device))

features = compute_embeddings(x, process_fn, batch_size=64, input_size=(150, 220))
data = (features, y, yforg)

exp_set = get_subset(data, range(0, 300))
dev_set = get_subset(data, range(300, 881))

rng = np.random.RandomState(1234)
_, results = run_fold(
    exp_set, dev_set,
    svm_type='rbf', C=1.0, gamma=2**-11,
    num_genuine_train=12, num_exp_neg=0, num_dev_neg=14,
    num_genuine_test=10, rng=rng,
)
print(results['all_metrics'])
```

---

## Available model architectures

| Key | Description |
|-----|-------------|
| `inkencoder` | Full model — 2048-dim embedding |
| `inkencoder_thin` | Lighter channel widths — 1024-dim embedding |
| `inkencoder_compact` | Fewer conv layers — 2048-dim embedding |

---

## Notes

- Input images are expected as grayscale, 8-bit (`uint8`).
- The preprocessing pipeline inverts pixel values (ink becomes bright) and centers
  the signature on a fixed canvas before resizing to 150×220.
- `state_dict` key names in saved checkpoints must not be changed if you want to
  load existing weights without remapping.
