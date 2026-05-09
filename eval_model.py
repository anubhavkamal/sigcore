"""
Evaluate an InkEncoder checkpoint on a signature verification dataset.

The dataset CSV must have lines:  path1,path2,label
where label=0 means genuine pair, label=1 means forgery pair.

Usage
-----
python eval_model.py \
    --model-path /path/to/model_best.pth \
    --data-dir  /path/to/test_images \
    --csv       /path/to/test_data.csv
"""

import sys
import os
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))

from inknet.encoder.model import InkEncoder
from inknet.preprocessing.transform import prepare_image
from skimage.io import imread
from skimage import img_as_ubyte


CANVAS_SIZE = (952, 1360)


def load_checkpoint(pth_path, device):
    state_dict, _, _ = torch.load(pth_path, map_location=device)
    model = InkEncoder().to(device).eval()
    model.load_state_dict(state_dict)
    return model


def load_image(img_path):
    img = img_as_ubyte(imread(img_path, as_gray=True))
    processed = prepare_image(img, CANVAS_SIZE)
    return torch.from_numpy(processed).float().div(255).view(1, 1, 150, 220)


def embed(model, tensor, device):
    with torch.no_grad():
        feat = model(tensor.to(device))
    return feat.cpu().numpy().flatten()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def equal_error_rate(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_checkpoint(args.model_path, device)
    print('Checkpoint loaded.\n')

    pairs = []
    with open(args.csv) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p1, p2, label = line.rsplit(',', 2)
            pairs.append((p1, p2, int(label)))

    print(f'Pairs: {len(pairs)}')

    y_true, scores, cache = [], [], {}

    for idx, (p1, p2, label) in enumerate(pairs, start=1):
        img1 = os.path.join(args.data_dir, p1)
        img2 = os.path.join(args.data_dir, p2)

        if img1 not in cache:
            cache[img1] = embed(model, load_image(img1), device)
        if img2 not in cache:
            cache[img2] = embed(model, load_image(img2), device)

        scores.append(cosine_similarity(cache[img1], cache[img2]))
        # label=0 → genuine pair (positive class); label=1 → forgery pair
        y_true.append(1 - label)

        if idx % 100 == 0 or idx == len(pairs):
            print(f'  processed {idx}/{len(pairs)}')

    auc = roc_auc_score(y_true, scores)
    eer, threshold = equal_error_rate(y_true, scores)

    print(f'\nAUC  : {auc:.4f}')
    print(f'EER  : {eer:.4f}')
    print(f'Threshold: {threshold:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate InkEncoder checkpoint')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--csv', required=True)
    main(parser.parse_args())
