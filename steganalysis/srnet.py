from __future__ import annotations

import argparse
import json
import math
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist
import wandb
from PIL import Image
from sklearn.metrics import accuracy_score, log_loss
from timm.models._manipulate import adapt_input_conv
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from utils.files import relative_stem, save_json, select_split_paths, split_relative_keys
from utils.reports import save_classification_outputs

DEFAULT_SRNET_CHECKPOINT_PATH = Path("assets/checkpoints/jin_srnet.ckpt")


@dataclass(slots=True)
class SRNetConfig:
    data_root: Path = Path("data")
    cover_dir_name: str = "cover"
    stego_dir_name: str = "stego"
    output_dir: Path = Path("runs/srnet")
    in_channels: int = 1
    num_classes: int = 2
    extensions: tuple[str, ...] = (".ppm",)
    epochs: int = 30
    batch_size: int = 8
    workers: int = 4
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_epochs: int = 3
    weight_decay: float = 0.
    seed: int = 1337
    fixed_val_suffix: str | None = "9"
    max_train_pairs: int | None = None
    max_val_pairs: int | None = None
    amp: bool = False
    compile: bool = True
    pretrained: bool = True
    pretrained_checkpoint: str | Path = DEFAULT_SRNET_CHECKPOINT_PATH
    strict_loading: bool = True
    wandb_project: str = "autostego-srnet"
    wandb_run_name: str | None = None
    wandb_mode: str = "offline"


CONFIG = SRNetConfig()
SRNET_CONFIG_JSON_ENV = "SRNET_CONFIG_JSON"


def config_from_dict(data: dict[str, Any]) -> SRNetConfig:
    values = dict(data)
    for key in ("data_root", "output_dir"):
        if key in values and values[key] is not None:
            values[key] = Path(values[key])

    if "pretrained_checkpoint" in values and values["pretrained_checkpoint"] is not None:
        checkpoint = values["pretrained_checkpoint"]
        values["pretrained_checkpoint"] = checkpoint if is_url(checkpoint) else Path(checkpoint)

    if "extensions" in values and values["extensions"] is not None:
        values["extensions"] = tuple(values["extensions"])

    return SRNetConfig(**values)


def load_config(path: str | Path) -> SRNetConfig:
    config_path = Path(path)
    return config_from_dict(json.loads(config_path.read_text(encoding="utf-8")))


def load_config_json(raw_config: str) -> SRNetConfig:
    return config_from_dict(json.loads(raw_config))


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def is_url(path_or_url: str | Path) -> bool:
    if isinstance(path_or_url, Path):
        return False
    parsed = urlparse(path_or_url)
    return parsed.scheme in {"http", "https"}


def load_srnet_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path = DEFAULT_SRNET_CHECKPOINT_PATH,
    *,
    strict_loading: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    if is_url(checkpoint_path):
        checkpoint = torch.hub.load_state_dict_from_url(str(checkpoint_path), map_location="cpu", progress=True)
    else:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"SRNet checkpoint not found at {checkpoint_file}. "
                "Generate it first with tests/scripts/remap_srnet_checkpoint.py."
            )
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    raw_state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(raw_state_dict, dict):
        raise TypeError("Unsupported checkpoint format for SRNet weights.")

    target_state = model.state_dict()
    loadable_state = dict(raw_state_dict)

    head_weight_key = "head.weight"
    head_bias_key = "head.bias"
    if head_weight_key in loadable_state and loadable_state[head_weight_key].shape != target_state[head_weight_key].shape:
        loadable_state.pop(head_weight_key, None)
        loadable_state.pop(head_bias_key, None)

    stem_weight_key = "features.0.0.weight"
    if stem_weight_key in loadable_state and loadable_state[stem_weight_key].shape != target_state[stem_weight_key].shape:
        loadable_state[stem_weight_key] = adapt_input_conv(target_state[stem_weight_key].shape[1], loadable_state[stem_weight_key])

    return model.load_state_dict(loadable_state, strict=strict_loading)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels),
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.skip(x)


class SRNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        *,
        pretrained: bool = False,
        checkpoint_path: str | Path = DEFAULT_SRNET_CHECKPOINT_PATH,
        strict_loading: bool = False,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 64),
            ConvBNAct(64, 16),
            *[ResidualBlock(16) for _ in range(5)],
            DownsampleBlock(16, 16),
            DownsampleBlock(16, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            ConvBNAct(256, 512),
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)
        if pretrained:
            load_srnet_checkpoint(self, checkpoint_path=checkpoint_path, strict_loading=strict_loading)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def load_grayscale_tensor(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def augment_pair_images(images: Tensor) -> Tensor:
    if random.random() < 0.5:
        images = torch.flip(images, dims=(-1,))
    rotations = random.randint(0, 3)
    if rotations:
        images = torch.rot90(images, k=rotations, dims=(-2, -1))
    return images


class PairDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, pairs: Sequence[CoverStegoPair], augment: bool = False) -> None:
        self.pairs = list(pairs)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        pair = self.pairs[index]
        images = torch.stack((load_grayscale_tensor(pair.cover), load_grayscale_tensor(pair.stego)), dim=0)
        if self.augment:
            images = augment_pair_images(images)
        labels = torch.tensor((0, 1), dtype=torch.long)
        return images, labels


@dataclass(slots=True)
class CoverStegoPair:
    cover: Path
    stego: Path


@dataclass(slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset: Dataset[object], *, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size


def setup_distributed() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(rank, local_rank, world_size, distributed, device)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(ctx: DistributedContext) -> bool:
    return ctx.rank == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_images(root: Path, extensions: Sequence[str]) -> list[Path]:
    allowed = {extension.lower() for extension in extensions}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def build_pairs(config: SRNetConfig) -> tuple[list[CoverStegoPair], Path]:
    cover_root = config.data_root / config.cover_dir_name
    stego_root = config.data_root / config.stego_dir_name
    if not cover_root.is_dir():
        raise FileNotFoundError(f"Cover directory not found: {cover_root}")
    if not stego_root.is_dir():
        raise FileNotFoundError(f"Stego directory not found: {stego_root}")

    cover_paths = collect_images(cover_root, config.extensions)
    if not cover_paths:
        raise FileNotFoundError(f"No cover images found in {cover_root}")

    stego_paths = collect_images(stego_root, config.extensions)
    if not stego_paths:
        raise FileNotFoundError(f"No stego images found in {stego_root}")

    stego_lookup = {relative_stem(path, stego_root): path for path in stego_paths}

    if config.fixed_val_suffix is not None:
        train_cover_paths, val_cover_paths = select_split_paths(
            cover_paths,
            cover_root,
            validation_suffix=config.fixed_val_suffix,
            max_train_items=config.max_train_pairs,
            max_val_items=config.max_val_pairs,
        )
        selected_cover_paths = train_cover_paths + val_cover_paths
    else:
        total_limit = None
        if config.max_train_pairs is not None or config.max_val_pairs is not None:
            total_limit = (config.max_train_pairs or 0) + (config.max_val_pairs or 0)
        selected_cover_paths = cover_paths[:total_limit] if total_limit else cover_paths

    if not selected_cover_paths:
        raise ValueError("No cover/stego pairs selected after applying split limits.")

    pairs: list[CoverStegoPair] = []
    for cover_path in selected_cover_paths:
        key = relative_stem(cover_path, cover_root)
        stego_path = stego_lookup.get(key)
        if stego_path is None:
            continue
        pairs.append(CoverStegoPair(cover=cover_path, stego=stego_path))

    if len(pairs) < 2:
        raise ValueError("Need at least 2 cover/stego pairs for training.")
    print(
        f"Image pairs surviving intersection: {len(pairs)} "
        f"(selected_cover={len(selected_cover_paths)}, stego={len(stego_paths)})"
    )
    return pairs, cover_root


def split_pairs(
    pairs: Sequence[CoverStegoPair],
    cover_root: Path,
    config: SRNetConfig,
) -> tuple[list[CoverStegoPair], list[CoverStegoPair]]:
    if config.fixed_val_suffix is None:
        raise ValueError("fixed_val_suffix must be set for SRNet training.")

    keyed_pairs = {relative_stem(pair.cover, cover_root): pair for pair in pairs}
    train_keys, val_keys = split_relative_keys(
        keyed_pairs,
        validation_suffix=config.fixed_val_suffix,
        max_train_items=config.max_train_pairs,
        max_val_items=config.max_val_pairs,
    )
    if not train_keys or not val_keys:
        raise ValueError("Fixed split needs both train and validation cover/stego pairs.")
    return [keyed_pairs[key] for key in train_keys], [keyed_pairs[key] for key in val_keys]


def build_dataloader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    workers: int,
    ctx: DistributedContext,
    shuffle: bool,
    drop_last: bool,
    pad_distributed: bool = True,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], Sampler[int] | None]:
    effective_batch_size = max(1, min(batch_size, len(dataset)))
    sampler: Sampler[int] | None = None
    if ctx.distributed:
        if pad_distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        else:
            sampler = DistributedEvalSampler(dataset, rank=ctx.rank, world_size=ctx.world_size)
    sampler_length = len(sampler) if sampler is not None else len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=sampler is None and shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=drop_last and sampler_length > effective_batch_size,
        collate_fn=pair_collate,
    )
    return loader, sampler


def pair_collate(batch: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    images = torch.cat([images for images, _ in batch], dim=0)
    labels = torch.cat([labels for _, labels in batch], dim=0)
    return images, labels


def maybe_all_reduce(values: Tensor, ctx: DistributedContext) -> Tensor:
    if ctx.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def gather_numpy(array: np.ndarray, ctx: DistributedContext) -> np.ndarray:
    if not ctx.distributed:
        return array
    gathered: list[np.ndarray | None] = [None] * ctx.world_size
    dist.all_gather_object(gathered, array)
    return np.concatenate([item for item in gathered if item is not None], axis=0)


def empty_numpy_batches(*, dtype: np.dtype[Any], width: int | None = None) -> np.ndarray:
    if width is None:
        return np.empty((0,), dtype=dtype)
    return np.empty((0, width), dtype=dtype)


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    ctx: DistributedContext,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
    collect_predictions: bool = False,
) -> tuple[float, float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    training = optimizer is not None
    model.train(training)

    loss_sum = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast_context(device, amp and device.type != "cpu"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        loss_sum += loss.detach() * batch_size
        correct += (predictions == labels).sum()
        count += batch_size
        if collect_predictions:
            all_labels.append(labels.detach().cpu().numpy())
            all_predictions.append(predictions.detach().cpu().numpy())
            all_probabilities.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())

    totals = maybe_all_reduce(torch.cat((loss_sum, correct, count)), ctx)
    if totals[2].item() == 0:
        raise ValueError("DataLoader produced zero samples for this epoch. Reduce batch_size or disable dropping the last batch.")
    mean_loss = (totals[0] / totals[2]).item()
    accuracy = (totals[1] / totals[2]).item()
    if not collect_predictions:
        return mean_loss, accuracy, None, None, None

    local_y_true = np.concatenate(all_labels, axis=0) if all_labels else empty_numpy_batches(dtype=np.int64)
    local_y_pred = np.concatenate(all_predictions, axis=0) if all_predictions else empty_numpy_batches(dtype=np.int64)
    local_y_prob = (
        np.concatenate(all_probabilities, axis=0)
        if all_probabilities
        else empty_numpy_batches(dtype=np.float32, width=num_classes)
    )

    y_true = gather_numpy(local_y_true, ctx)
    y_pred = gather_numpy(local_y_pred, ctx)
    y_prob = gather_numpy(local_y_prob, ctx)
    return mean_loss, accuracy, y_true, y_pred, y_prob


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_acc: float,
    is_best: bool,
    *,
    config: SRNetConfig,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    train_pairs: int,
    val_pairs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint = {
        "epoch": epoch,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, output_dir / "last.pt")
    save_validation_outputs(config, y_true, y_pred, y_prob, train_pairs, val_pairs, label="last")
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")
        save_validation_outputs(config, y_true, y_pred, y_prob, train_pairs, val_pairs, label="best")


def build_model(config: SRNetConfig, device: torch.device) -> nn.Module:
    model = SRNet(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        checkpoint_path=config.pretrained_checkpoint,
        strict_loading=config.strict_loading,
    )
    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model.to(device)


def init_wandb(config: SRNetConfig, ctx: DistributedContext) -> None:
    if not is_main_process(ctx):
        return
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config={key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    )


def finish_wandb(ctx: DistributedContext) -> None:
    if is_main_process(ctx) and wandb.run is not None:
        wandb.finish()


def save_validation_outputs(
    config: SRNetConfig,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    train_pairs: int,
    val_pairs: int,
    *,
    label: str = "last",
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_images": train_pairs * 2,
        "val_images": val_pairs * 2,
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_log_loss": float(log_loss(y_true, y_prob)),
    }
    metrics_name = "metrics.json" if label == "last" else f"metrics_{label}.json"
    report_stem = "classification_report" if label == "last" else f"classification_report_{label}"
    save_json(config.output_dir / metrics_name, metrics)
    save_json(config.output_dir / "train_config.json", asdict(config))
    save_classification_outputs(config.output_dir, y_true, y_pred, report_stem=report_stem)


def lr_for_epoch(config: SRNetConfig, epoch: int) -> float:
    if config.epochs <= 1:
        return config.lr

    warmup_epochs = min(config.warmup_epochs, config.epochs)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        warmup_progress = epoch / warmup_epochs
        return config.min_lr + (config.lr - config.min_lr) * warmup_progress

    decay_steps = max(1, config.epochs - warmup_epochs)
    decay_progress = (epoch - warmup_epochs) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return config.min_lr + (config.lr - config.min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(config: SRNetConfig = CONFIG) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    ctx = setup_distributed()
    seed_everything(config.seed + ctx.rank)

    try:
        init_wandb(config, ctx)
        pairs, cover_root = build_pairs(config)
        train_pairs, val_pairs = split_pairs(pairs, cover_root, config)

        train_dataset = PairDataset(train_pairs, augment=True)
        val_dataset = PairDataset(val_pairs, augment=False)
        train_loader, train_sampler = build_dataloader(
            train_dataset,
            config.batch_size,
            config.workers,
            ctx,
            shuffle=True,
            drop_last=True,
            pad_distributed=True,
        )
        val_loader, _ = build_dataloader(
            val_dataset,
            config.batch_size,
            config.workers,
            ctx,
            shuffle=False,
            drop_last=False,
            pad_distributed=False,
        )

        model = build_model(config, ctx.device)
        if ctx.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        best_val_acc = -1.0
        if is_main_process(ctx):
            print(f"train pairs: {len(train_pairs)} | val pairs: {len(val_pairs)}")
            print(
                "images per optimization step: "
                f"{min(config.batch_size, len(train_dataset)) * 2 * ctx.world_size}"
            )
            print(f"cover dir: {config.cover_dir_name} | stego dir: {config.stego_dir_name}")

        for epoch in range(1, config.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            current_lr = lr_for_epoch(config, epoch)
            set_optimizer_lr(optimizer, current_lr)

            train_loss, train_acc, _, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                ctx.device,
                ctx,
                config.num_classes,
                optimizer=optimizer,
                amp=config.amp,
            )
            with torch.inference_mode():
                val_loss, val_acc, y_true, y_pred, y_prob = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    ctx.device,
                    ctx,
                    config.num_classes,
                    optimizer=None,
                    amp=config.amp,
                    collect_predictions=True,
                )

            if is_main_process(ctx):
                print(
                    f"epoch {epoch:03d} | "
                    f"lr={current_lr:.6g} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "lr": current_lr,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                    },
                    step=epoch,
                )
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                save_checkpoint(
                    config.output_dir,
                    epoch,
                    model,
                    optimizer,
                    best_val_acc,
                    is_best,
                    config=config,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    train_pairs=len(train_pairs),
                    val_pairs=len(val_pairs),
                )
    finally:
        finish_wandb(ctx)
        cleanup_distributed(ctx)


def detect(config: SRNetConfig = CONFIG) -> None:
    train(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SRNet steganalysis model")
    parser.add_argument("--config", type=Path, help="Path to a JSON-serialized SRNetConfig", default=None)
    args = parser.parse_args()
    if args.config is not None:
        config = load_config(args.config)
    elif SRNET_CONFIG_JSON_ENV in os.environ:
        config = load_config_json(os.environ[SRNET_CONFIG_JSON_ENV])
    else:
        config = CONFIG
    detect(config)


if __name__ == "__main__":
    main()