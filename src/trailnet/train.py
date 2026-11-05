"""Train TrailNet using geographic-block train/validation/test partitions."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trailnet.config import load_config, resolve_path
from trailnet.data.dataset import TrailTileDataset, read_indexes, spatial_split
from trailnet.models.unet import TrailUNet, dice_bce_loss


def metrics(logits, target, threshold=0.5):
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    union = tp + fp + fn
    return (tp / union if union else 1.0,
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0)


def run_training(config_path=None, area_names=None):
    config = load_config(config_path)
    if area_names is not None:
        config["areas"] = [area for area in config["areas"] if area["name"] in area_names]
        if not config["areas"]:
            raise ValueError("No configured areas selected for training")
    train_cfg = config["training"]
    seed = int(config.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    indexes = [resolve_path(config, config["paths"]["tile_dir"]) / a["name"] / "index.csv"
               for a in config["areas"]]
    missing = [p for p in indexes if not p.exists()]
    if missing:
        raise FileNotFoundError("Prepare tiles first; missing indexes: " + ", ".join(map(str, missing)))
    rows = read_indexes(indexes)
    splits = spatial_split(rows, train_cfg["block_size_m"], train_cfg["val_fraction"],
                           train_cfg["test_fraction"], seed)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Spatial split produced an empty train or validation set; adjust block size/fractions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = TrailUNet(model_cfg["encoder_name"], model_cfg.get("encoder_weights"),
                      model_cfg["in_channels"], model_cfg["classes"],
                      model_cfg.get("dilated_center", False)).to(device)
    loaders = {
        "train": DataLoader(TrailTileDataset(splits["train"], True), batch_size=train_cfg["batch_size"],
                            shuffle=True, num_workers=train_cfg.get("num_workers", 0)),
        "val": DataLoader(TrailTileDataset(splits["val"]), batch_size=train_cfg["batch_size"],
                          shuffle=False, num_workers=train_cfg.get("num_workers", 0)),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    amp = bool(train_cfg.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint = resolve_path(config, train_cfg["checkpoint_path"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    for epoch in range(int(train_cfg["epochs"])):
        summary = {}
        for phase in ("train", "val"):
            model.train(phase == "train")
            loss_sum = iou_sum = f1_sum = count = 0
            for images, masks in tqdm(loaders[phase], desc=f"{phase} {epoch + 1}", leave=False):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device_type=device.type, enabled=amp):
                        logits = model(images)
                        loss = dice_bce_loss(logits, masks)
                    if phase == "train":
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                iou, f1 = metrics(logits.detach(), masks)
                loss_sum += loss.item(); iou_sum += iou; f1_sum += f1; count += 1
            summary[phase] = (loss_sum / count, iou_sum / count, f1_sum / count)
        print(f"epoch={epoch+1} train_loss={summary['train'][0]:.4f} "
              f"val_loss={summary['val'][0]:.4f} val_iou={summary['val'][1]:.4f} val_f1={summary['val'][2]:.4f}")
        if summary["val"][2] > best_f1:
            best_f1 = summary["val"][2]
            torch.save({"model": model.state_dict(), "config": model_cfg,
                        "epoch": epoch + 1, "val_f1": best_f1, "seed": seed}, checkpoint)
    print(f"Best validation F1: {best_f1:.4f}; checkpoint: {checkpoint}")
    print(f"Held-out test blocks reserved: {len(splits['test'])} tiles")
    return checkpoint
