#!/usr/bin/env python3
"""
Efficient MLP training from a huge .pt file using memory-mapped tensor storages.

Works well when the .pt contains:
  - dict: {"x": X, "y": Y}  (preferred)
  - tuple/list: (X, Y)

Where X and Y are torch.Tensors with first dim = N samples.

Key idea: torch.load(..., mmap=True) memory-maps tensor storages to avoid loading
all tensor bytes into RAM at once. (Still reads metadata.) See PyTorch docs.

Notes:
- For very large N, avoid DataLoader(shuffle=True) which may create a full randperm.
  This script uses sampling-with-replacement via an IterableDataset so memory stays flat.
"""

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from PDF_net import PDFHR_Adapter


def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,  
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


class RandomIndexIterableDataset(IterableDataset):
    """
    Streams (x, y) pairs by sampling random indices with replacement.
    This avoids building a full permutation (randperm) for huge datasets.
    """
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, steps_per_epoch: int, batch_size: int, seed: int = 0):
        super().__init__()
        assert X.shape[0] == Y.shape[0], "X and Y must have same number of rows"
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.steps_per_epoch = int(steps_per_epoch)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        g = torch.Generator()
        g.manual_seed(self.seed + 1337 * worker_id)

        batches_total = self.steps_per_epoch
        batches_for_this_worker = (batches_total + num_workers - 1) // num_workers

        for _ in range(batches_for_this_worker):
            idx = torch.randint(0, self.n, (self.batch_size,), generator=g)
            x = self.X[idx]
            y = self.Y[idx]
            yield x, y


def load_xy_mmap(path: str, x_key: str = "x", y_key: str = "y"):
    """
    Loads a .pt file with mmap=True and returns (X, Y) CPU tensors.
    """
    obj = torch.load(path, map_location="cpu", mmap=True)

    if isinstance(obj, dict) and (x_key in obj) and (y_key in obj):
        X, Y = obj[x_key], obj[y_key]
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        X, Y = obj[0], obj[1]
    else:
        raise TypeError(
            "Unsupported .pt structure.\n"
            "Expected dict with keys {x_key,y_key} or a 2-tuple/list (X,Y).\n"
            "If your file is a Python list of (x,y) samples, it can't be streamed efficiently.\n"
            "Convert it to big tensors or chunked shards first."
        )

    if not (isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor)):
        raise TypeError("X and Y must be torch.Tensors for efficient mmap training.")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y have different lengths: {X.shape[0]} vs {Y.shape[0]}")

    return X, Y


@dataclass
class TrainCfg:
    data: str
    input_dim: int
    output_dim: int
    hidden_dim: int
    depth: int
    dropout: float
    batch_size: int
    steps_per_epoch: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    device: str
    amp: bool
    compile: bool
    seed: int
    ckpt_dir: str


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default='../precomputed_data/sampling_pose_L1.pt'
, 
                    help="Path to .pt containing (X,Y) tensors or {'x':X,'y':Y}")
    p.add_argument("--x-key", default="db")
    p.add_argument("--y-key", default="dis")

    p.add_argument("--input-dim", type=int, default=29)
    p.add_argument("--output-dim", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--steps-per-epoch", type=int, default=300, help="Batches per epoch (sampling with replacement)")
    p.add_argument("--epochs", type=int, default=50)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--no-pin-memory", action="store_true")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile", action="store_true", help="Use torch.compile when available")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="../prior_ckpts")

    args = p.parse_args()

    cfg = TrainCfg(
        data=args.data,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.persistent_workers),
        pin_memory=not args.no_pin_memory,
        device=args.device,
        amp=not args.no_amp,
        compile=bool(args.compile),
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    set_seed(cfg.seed)

    # Speed knobs (safe defaults)
    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading mmap tensors from: {cfg.data}")
    X, Y = load_xy_mmap(cfg.data, x_key=args.x_key, y_key=args.y_key)
    print(f"Loaded: X={tuple(X.shape)} {X.dtype}, Y={tuple(Y.shape)} {Y.dtype}")

    if X.ndim != 2:
        print("Warning: X is not 2D. Make sure your MLP input_dim matches X.shape[-1].")
    if X.shape[-1] != cfg.input_dim:
        raise ValueError(f"input_dim={cfg.input_dim} but X.shape[-1]={X.shape[-1]}")
    if Y.shape[-1] != cfg.output_dim and not (cfg.output_dim == 1 and Y.ndim == 1):
        print("Warning: Y last-dim doesn't match output_dim. Adjust --output-dim or reshape Y.")

    # Ensure float types for MLP regression
    if not torch.is_floating_point(X):
        X = X.float()
    if not torch.is_floating_point(Y):
        Y = Y.float()
    if Y.ndim == 1 and cfg.output_dim == 1:
        Y = Y.view(-1, 1)

    dataset = RandomIndexIterableDataset(X, Y, cfg.steps_per_epoch, cfg.batch_size, seed=cfg.seed)

    loader = DataLoader(
        dataset,
        batch_size=None,  # dataset already yields batches
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and cfg.device.startswith("cuda"),
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    model = PDFHR_Adapter(device=cfg.device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Updated to modern PyTorch syntax
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and cfg.device.startswith("cuda")))

    global_step = 0
    eikonal_weight = 1  # Kept as a static variable based on your logic

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0

        for step, (xb, yb) in enumerate(loader):
            if step >= cfg.steps_per_epoch:
                break

            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True).squeeze(-1)

            optimizer.zero_grad(set_to_none=True)
            xb.requires_grad_(True)

            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            if eikonal_weight > 0:
                gradients = get_gradient(xb, pred)
                grad_norm = gradients.norm(2, dim=1) 
                eikonal_loss_val = ((grad_norm - 1) ** 2).mean()
                total_loss = loss + eikonal_weight * eikonal_loss_val
            else:
                total_loss = loss

            total_loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            global_step += 1

            if (step + 1) % 100 == 0:
                avg = running / 100
                running = 0.0
                dt = time.time() - t0
                it_s = (step + 1) / max(dt, 1e-9)
                print(f"epoch {epoch+1}/{cfg.epochs} step {step+1}/{cfg.steps_per_epoch} "
                      f"loss={avg:.6g} it/s={it_s:.2f}")

        # Save checkpoint
        ckpt_path = os.path.join(cfg.ckpt_dir, f"mlp_latest_1_epoch{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()