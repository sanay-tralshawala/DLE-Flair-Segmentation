"""
Shared training loop for FLAIR segmentation models.

Primary entry point: `train_from_config(<model_config_path>)`
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src.data import build_dataloaders
from src.models import build_model, freeze_backbone, unfreeze_backbone
from src.evaluate import evaluate_model
from src.utils import load_config, get_device


def build_loss(loss_config: dict, device: torch.device) -> nn.Module:
    """
    Build the segmentation loss function.

    Two options:
        - "cross_entropy": standard cross-entropy loss for multi-class segmentation
        - "weight_cross_entropy": cross-entropy loss with class weights to handle class imbalance

    Expected logits: [batch_size, num_classes, height, width]
    Expected targets: [batch_size, height, width] with class indices in [0, num_classes-1]
    """
    loss_name = loss_config["name"]
    ignore_index = loss_config.get("ignore_index", 255)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    if loss_name == "weight_cross_entropy":
        class_weights = torch.tensor(loss_config["class_weights"], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    raise ValueError(f"Unknown loss: {loss_name}")


def _backbone_parameters(model: nn.Module) -> list[nn.Parameter]:
    if hasattr(model, "backbone"):
        return list(model.backbone.parameters())
    if hasattr(model, "encoder"):
        return list(model.encoder.parameters())
    raise AttributeError("Model must expose a `backbone` or `encoder` attribute.")


def _head_parameters(model: nn.Module) -> list[nn.Parameter]:
    backbone_param_ids = {id(param) for param in _backbone_parameters(model)}
    return [
        param
        for param in model.parameters()
        if id(param) not in backbone_param_ids and param.requires_grad
    ]


def build_optimizer(model: nn.Module, optimizer_config: dict) -> optim.Optimizer:
    """
    Build the optimizer using a lower LR for backbone params and a higher LR for head params.

    Supported optimizers:
        - "adam"
        - "adamw"
    """
    optimizer_name = optimizer_config["name"]
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    param_groups = []
    backbone_params = [param for param in _backbone_parameters(model) if param.requires_grad]
    head_params = _head_parameters(model)

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": optimizer_config["backbone_lr"],
            "name": "backbone",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": optimizer_config["head_lr"],
            "name": "head",
        })

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    if optimizer_name == "adam":
        return optim.Adam(param_groups, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(param_groups, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: dict,
    max_epochs: int,
) -> optim.lr_scheduler.LRScheduler | None:
    if not scheduler_config.get("enabled", False):
        return None

    scheduler_name = scheduler_config["name"]
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def is_better(current: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unknown comparison mode: {mode}")


def _unpack_batch(batch: dict | tuple) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"], batch["mask"]
    return batch


def train_one_epoch(
    model: nn.Module,
    train_loader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Train the model for one epoch and return the average training loss.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        images, masks = _unpack_batch(batch)
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=loss.item())

    if num_batches == 0:
        raise ValueError("Training dataloader is empty.")

    return {"train_loss": running_loss / num_batches}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
) -> dict:
    """
    Validate for one epoch. Metrics beyond val_loss come from evaluate.py modules.
    """
    model.eval()
    return evaluate_model(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    config: dict,
    metrics: dict,
    best_metric: float | None,
) -> None:
    """
    Save the model checkpoint to local disk.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "metrics": metrics,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, checkpoint_path)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _current_lrs(optimizer: optim.Optimizer) -> dict:
    return {
        f"{group.get('name', f'group_{index}')}_lr": group["lr"]
        for index, group in enumerate(optimizer.param_groups)
    }


def _load_env_file() -> None:
    """
    Load local environment variables for notebook/script runs.
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        return
    except ModuleNotFoundError:
        # if dotenv is not installed, notify but dont fail, might use colab secrets
        print("python-dotenv not found, skipping .env loading. OK if using Colab secrets for W&B API key.")


def _init_wandb(config: dict):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    import wandb

    _load_env_file()
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "W&B is enabled, but WANDB_API_KEY is not set. "
            "For local runs, add WANDB_API_KEY to repo-root .env. "
            "For Colab runs, add a Colab secret named WANDB_API_KEY before training."
        )
    wandb.login(key=api_key)

    return wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        group=wandb_config.get("group"),
        job_type=wandb_config.get("job_type", "train"),
        name=config["experiment"]["name"],
        tags=wandb_config.get("tags", []),
        config=config,
    )


def _log_wandb_checkpoints(run, checkpoint_paths: list[Path], artifact_name: str) -> None:
    import wandb

    artifact = wandb.Artifact(artifact_name, type="model")
    for checkpoint_path in checkpoint_paths:
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact)


def finish_wandb_run(run) -> None:
    """Finish a W&B run when notebook-side logging is complete."""
    if run is not None:
        run.finish()


def train_from_config(config_path: str | Path) -> dict:
    """
    Main training loop that trains a FLAIR segmentation model from a merged config.
    """
    config = load_config(config_path)
    _set_seed(config["experiment"]["seed"])

    device = get_device()
    output_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(config["data"])
    model = build_model(config["model"]).to(device)
    config["training"]["loss"].setdefault("ignore_index", config["data"].get("ignore_index", 255))
    loss_fn = build_loss(config["training"]["loss"], device)

    warmup_frozen_epochs = config["training"].get("warmup_frozen_epochs", 0)
    if warmup_frozen_epochs > 0:
        freeze_backbone(model)

    optimizer = build_optimizer(model, config["training"]["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        config["training"]["scheduler"],
        config["training"]["max_epochs"],
    )

    checkpoint_config = config["checkpoint"]
    early_stopping_config = config["training"]["early_stopping"]
    monitor = checkpoint_config["monitor"]
    mode = checkpoint_config["mode"]
    min_delta = early_stopping_config.get("min_delta", 0.0)

    best_metric = None
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    run = _init_wandb(config)

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        if epoch == warmup_frozen_epochs + 1 and warmup_frozen_epochs > 0:
            unfreeze_backbone(model)
            optimizer = build_optimizer(model, config["training"]["optimizer"])
            scheduler = build_scheduler(
                optimizer,
                config["training"]["scheduler"],
                config["training"]["max_epochs"] - epoch + 1,
            )

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            num_classes=config["data"]["num_classes"],
            ignore_index=config["data"].get("ignore_index", 255),
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            **_current_lrs(optimizer),
        }
        history.append(metrics)

        current_metric = metrics[monitor]
        improved = is_better(current_metric, best_metric, mode, min_delta)
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if checkpoint_config.get("save_last", True):
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                metrics,
                best_metric,
            )

        if improved and checkpoint_config.get("save_best", True):
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                metrics,
                best_metric,
            )

        if run is not None:
            run.log(metrics, step=epoch)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_IoU={val_metrics.get('val_IoU', float('nan')):.4f} "
            f"{monitor}={current_metric:.4f}"
        )

        can_stop = epoch >= config["training"]["min_epochs"]
        should_stop = (
            early_stopping_config.get("enabled", True)
            and can_stop
            and epochs_without_improvement >= early_stopping_config["patience"]
        )
        if should_stop:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    if run is not None and config.get("wandb", {}).get("log_model", True):
        checkpoint_paths = [
            path
            for path in (output_dir / "best.pt", output_dir / "last.pt")
            if path.exists()
        ]
        if checkpoint_paths:
            _log_wandb_checkpoints(
                run,
                checkpoint_paths,
                artifact_name=f"{config['experiment']['name']}-checkpoints",
            )

    return {
        "output_dir": output_dir,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history,
        "run": run,
    }
