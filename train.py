"""
Train the InkEncoder feature extractor on pre-processed signature data.

Usage
-----
python train.py \
    --dataset-path /path/to/data.npz \
    --model inkencoder \
    --logdir ./runs/exp1 \
    --epochs 60 \
    --lr 0.001

The dataset .npz must contain arrays: x (N,1,H,W), y (N,), yforg (N,).
Use inknet/datasets/util.py::build_dataset() to create one from raw images.
"""

import sys
import os
import argparse
import pathlib
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))

import inknet.encoder as models
from inknet.encoder.loader import AugmentedDataset
import inknet.datasets.util as util


# ---------------------------------------------------------------------------
# Forward / backward helpers
# ---------------------------------------------------------------------------

def _forward_pass(base_model, cls_head, forg_head, batch, device, args):
    x = batch[0].float().to(device)
    y = batch[1].long().to(device)
    yforg = batch[2].float().to(device)

    feats = base_model(x)

    if args.forg:
        if args.loss_type == 'L1':
            logits = cls_head(feats)
            cls_loss = F.cross_entropy(logits, y)
            forg_logits = forg_head(feats).squeeze()
            forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)
            loss = (1 - args.lamb) * cls_loss + args.lamb * forg_loss
        else:
            mask = yforg == 0
            cls_loss = F.cross_entropy(cls_head(feats[mask]), y[mask]) if mask.any() else torch.tensor(0.0)
            forg_logits = forg_head(feats).squeeze()
            forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)
            loss = (1 - args.lamb) * cls_loss + args.lamb * forg_loss
    else:
        logits = cls_head(feats)
        loss = cls_loss = F.cross_entropy(logits, y)
        forg_loss = torch.tensor(0.0)

    return loss, cls_loss, forg_loss


def _train_one_epoch(loader, base_model, cls_head, forg_head,
                     optimizer, lr_scheduler, device, epoch, args):
    base_model.train()
    cls_head.train()
    forg_head.train()

    for step, batch in enumerate(loader):
        loss, cls_loss, _ = _forward_pass(base_model, cls_head, forg_head, batch, device, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'], 10)
        optimizer.step()

        if step % 50 == 0:
            print(f'  epoch {epoch:3d} step {step:4d}/{len(loader)}  loss={loss.item():.4f}  '
                  f'cls={cls_loss.item():.4f}')

    lr_scheduler.step()


@torch.no_grad()
def _validate(loader, base_model, cls_head, forg_head, device, args):
    base_model.eval()
    cls_head.eval()
    losses, accs = [], []

    for batch in loader:
        x = batch[0].float().to(device)
        y = batch[1].long().to(device)
        yforg = batch[2].float().to(device)

        feats = base_model(x)
        mask = yforg == 0
        logits = cls_head(feats[mask])
        loss = F.cross_entropy(logits, y[mask])
        acc = y[mask].eq(logits.argmax(1)).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())

    return float(np.mean(accs)), float(np.mean(losses))


def _get_params(base_model, cls_head, forg_head):
    def _cpu(sd):
        return OrderedDict((k, v.cpu()) for k, v in sd.items())
    return _cpu(base_model.state_dict()), _cpu(cls_head.state_dict()), _cpu(forg_head.state_dict())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_loaders(data, batch_size, input_size):
    le = LabelEncoder()
    y = le.fit_transform(data[1])
    ds = TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(y), torch.from_numpy(data[2]))
    n_train = int(0.9 * len(ds))
    train_set, val_set = random_split(ds, [n_train, len(ds) - n_train])

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(AugmentedDataset(train_set, train_tf), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AugmentedDataset(val_set, val_tf), batch_size=batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    device = (torch.device('cuda', args.gpu_idx) if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'Device: {device}')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print('Loading dataset …')
    x, y, yforg, _, _ = util.load_dataset(args.dataset_path)
    data = util.get_subset((x, y, yforg), range(*args.users))
    if not args.forg:
        data = util.remove_forgeries(data)

    train_loader, val_loader = build_loaders(data, args.batch_size, tuple(args.input_size))

    n_classes = len(np.unique(data[1]))
    print(f'Classes: {n_classes}')

    base_model = models.available_models[args.model]().to(device)
    cls_head = nn.Linear(base_model.feature_space_size, n_classes).to(device)
    forg_head = (nn.Linear(base_model.feature_space_size, 1).to(device)
                 if args.forg else nn.Module().to(device))

    if args.checkpoint:
        sd, _, _ = torch.load(args.checkpoint, map_location=device)
        base_model.load_state_dict(sd)
        print('Loaded checkpoint:', args.checkpoint)

    params = list(base_model.parameters()) + list(cls_head.parameters())
    if args.forg:
        params += list(forg_head.parameters())

    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                          nesterov=True, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.epochs // args.lr_decay_times, args.lr_decay)

    best_acc = 0.0
    best_params = _get_params(base_model, cls_head, forg_head)

    for epoch in range(args.epochs):
        _train_one_epoch(train_loader, base_model, cls_head, forg_head,
                         optimizer, scheduler, device, epoch, args)
        val_acc, val_loss = _validate(val_loader, base_model, cls_head, forg_head, device, args)
        print(f'Epoch {epoch:3d}  val_loss={val_loss:.4f}  val_acc={val_acc * 100:.2f}%')

        if val_acc >= best_acc:
            best_acc = val_acc
            best_params = _get_params(base_model, cls_head, forg_head)
            torch.save(best_params, logdir / 'model_best.pth')

        torch.save(_get_params(base_model, cls_head, forg_head), logdir / 'model_last.pth')

    print(f'Training complete. Best val acc: {best_acc * 100:.2f}%')
    print(f'Checkpoints saved to: {logdir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train InkEncoder')
    ap.add_argument('--dataset-path', required=True, help='.npz dataset file')
    ap.add_argument('--model', choices=list(models.available_models.keys()),
                    default='inkencoder')
    ap.add_argument('--logdir', required=True, help='Output directory for checkpoints')
    ap.add_argument('--input-size', nargs=2, type=int, default=[150, 220])
    ap.add_argument('--users', nargs=2, type=int, default=[350, 881])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--lr-decay', type=float, default=0.1)
    ap.add_argument('--lr-decay-times', type=int, default=3)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--forg', action='store_true', default=False)
    ap.add_argument('--lamb', type=float, default=0.95,
                    help='Weight for forgery loss when --forg is set')
    ap.add_argument('--loss-type', choices=['L1', 'L2'], default='L2')
    ap.add_argument('--checkpoint', help='Path to a .pth to resume from')
    ap.add_argument('--gpu-idx', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()
    print(args)
    main(args)
